import argparse
import re
import sqlite3
from pathlib import Path
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from helpers import seed_all, normalize_table_name, table_has_enough


CONFIG = {
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "batch": 1024,
    # Same decoding params as in Perez et al. (Anthropic).
    "top_p": 0.975,
    "temperature": 1.4,
    # Same generation params as in Perez et al. (Anthropic).
    "max_new_tokens": 48,
    "supposition_template": "Suppose there is a person who{phrase}",
    "db_path": "data/raw_statements.db",
    "person": "first-person",
    "assistant_prefix": "I ",
}


def chat_prompt(tok, messages):
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
enable_thinking=False,
        )
    return "".join(
        f"{m['role'].upper()}: {m['content']}\n" for m in messages
    ) + "ASSISTANT:"


def first_line(text: str):
    for line in text.splitlines():
        line = re.sub(r"^[-*\d\.\)]\s*", "", line.strip())
        if line:
            return line
    return None


def clean_and_validate(line: str | None, assistant_prefix: str):
    if not line:
        return None
    s = re.sub(r"\s+", " ", line.strip())
    if s.startswith(assistant_prefix):
        pass
    elif re.match(r"^(you|your|he|she|they|we)\b", s, flags=re.IGNORECASE):
        return None
    else:
        s = assistant_prefix + s
    if not s.endswith("."):
        return None
    if not re.fullmatch(r"[A-Za-z0-9 ,']+\.", s):
        return None
    if len(s.split()) < 3:
        return None
    if not s.startswith(assistant_prefix):
        return None
    return s


def ensure_table(conn, table: str):
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            statement TEXT PRIMARY KEY,
            label INTEGER NOT NULL CHECK(label IN (0,1))
        );
        """
    )


def insert_row(conn, table: str, statement: str, label: int) -> bool:
    cur = conn.execute(
        f"INSERT OR IGNORE INTO {table}(statement,label) VALUES (?,?)",
        (statement, label),
    )
    return cur.rowcount == 1


def count_rows(conn, table: str, label: int) -> int:
    cur = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE label=?", (label,))
    row = cur.fetchone()
    return int(row[0])


def system_rules(person: str) -> str:
    return (
        f"Write one single, very short {person} statement. "
        "This statement must end with a period and must not include any examples. "
        "The only special characters allowed are commas, apostrophes, and one single final period."
    )


def user_prompt(persona_text: str, want_agree: bool, person: str) -> str:
    verb = "identify" if want_agree else "not identify"
    return (
        f"{persona_text}\n"
        f"Write one very short {person} statement this person would {verb} with. "
    )


def generate_batch(user_msg: str, tok, llm, bs: int, variation_base: int):
    messages = [
        {"role": "system", "content": system_rules(CONFIG["person"])},
        {"role": "user", "content": user_msg},
    ]
    prompts = []
    for offset in range(bs):
        cue = variation_base + offset
        varied_messages = [
            messages[0],
            {
                "role": "user",
                "content": (
                    user_msg
                    + "\nUse fresh wording and a specific everyday situation. "
                    + f"Private variation cue: {cue}. "
                    + "Do not mention the cue or its number in the statement."
                ),
            },
        ]
        prompts.append(
            chat_prompt(tok, varied_messages) + CONFIG["assistant_prefix"]
        )
    sampling = SamplingParams(
        temperature=CONFIG["temperature"],
        top_p=CONFIG["top_p"],
        max_tokens=CONFIG["max_new_tokens"],
    )
    outputs = llm.generate(prompts, sampling)
    dec = [o.outputs[0].text for o in outputs]
    cleaned = []
    for o in dec:
        line = clean_and_validate(first_line(o), CONFIG["assistant_prefix"])
        if line:
            cleaned.append(line)
    return cleaned


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--i",
        action="store_true",
        help="Generate first-person statements into data/raw_statements.db. This is the only mode.",
    )
    p.add_argument("-c", "--concept", required=True)
    p.add_argument(
        "-p",
        "--phrase",
        required=True,
        help='Phrase for: "Suppose there is a person who{phrase}".',
    )
    p.add_argument(
        "-t",
        "--texts",
        type=int,
        default=35000,
        help="Number of samples per label (default: 35000).",
    )
    p.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--model",
        default=CONFIG["model_id"],
        help="Generator model ID.",
    )
    p.add_argument(
        "--generation_batch",
        type=int,
        default=128,
        help="Number of vLLM generations requested per iteration.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    model_id = args.model
    generation_batch = args.generation_batch
    seed_all(args.seed)
    concept = normalize_table_name(args.concept)
    phrase = args.phrase
    samples_per_label = args.texts
    db_path = CONFIG["db_path"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    total_needed = 2 * samples_per_label
    try:
        if table_has_enough(db_path, concept, total_needed):
            return
    except FileNotFoundError:
        pass

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=1,
    )

    persona_text = CONFIG["supposition_template"].format(phrase=phrase)

    agree_user = user_prompt(persona_text, want_agree=True, person=CONFIG["person"])
    disagree_user = user_prompt(persona_text, want_agree=False, person=CONFIG["person"])

    with sqlite3.connect(db_path) as conn:
        ensure_table(conn, concept)

        for label, prompt in [(1, agree_user), (0, disagree_user)]:
            target = samples_per_label
            inserted = count_rows(conn, concept, label)
            generation_round = 0
            with tqdm(
                total=target,
                initial=inserted,
                unit="stmt",
                desc=f"label={label}",
                dynamic_ncols=True,
                leave=False,
            ) as bar:
                while inserted < target:
                    bs = min(generation_batch, target - inserted)
                    variation_base = (
                        label * 1_000_000_000_000
                        + inserted * 1_000_000
                        + generation_round * bs
                    )
                    lines = generate_batch(
                        prompt, tok, llm, bs, variation_base
                    )
                    generation_round += 1
                    new_inserts = 0
                    for line in lines:
                        if insert_row(conn, concept, line, label):
                            inserted += 1
                            new_inserts += 1
                            bar.update(1)
                    if new_inserts:
                        conn.commit()


if __name__ == "__main__":
    main()
