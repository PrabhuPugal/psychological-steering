"""Provider-level smoke test for OSWorld V1's new automated (clean-boot-per-task)
qemu provider -- analogous to the one already validated for WeaveBench, but for
OSWorld V1's own desktop_env/providers/qemu/ using this tree's own Ubuntu.qcow2.
No model, no task classes needed -- just proves provider+VM+firmware+ports work.
"""
import sys

sys.path.insert(0, ".")
from desktop_env.providers import create_vm_manager_and_provider

out_path = sys.argv[1] if len(sys.argv) > 1 else "osworld_v1_qemu_smoke_screenshot.png"

manager, provider = create_vm_manager_and_provider("qemu", region=None, use_proxy=False)
vm_path = manager.get_vm_path()
print(f"Booting {vm_path} under QEMU (TCG, no KVM on this cluster)...")

provider.start_emulator(vm_path, headless=True, os_type="Ubuntu")
print("Guest agent is up. Fetching a screenshot...")

import requests

ip = provider.get_ip_address(vm_path)
server_port = ip.split(":")[1]
resp = requests.get(f"http://localhost:{server_port}/screenshot", timeout=30)
resp.raise_for_status()
with open(out_path, "wb") as f:
    f.write(resp.content)
print(f"Saved {len(resp.content)} bytes to {out_path}")

cmd_resp = requests.post(
    f"http://localhost:{server_port}/execute",
    json={"command": ["bash", "-c", "echo osworld-v1-qemu-smoke-ok"], "shell": False},
    timeout=30,
)
print("Guest command exit:", cmd_resp.status_code, cmd_resp.text[:500])

provider.stop_emulator(vm_path)
print("VM stopped cleanly. Smoke test PASSED.")
