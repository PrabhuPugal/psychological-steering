"""Docker-free provider: runs the OSWorld VM directly under QEMU.

Meant for hosts (e.g. shared HPC clusters) that have neither Docker, VMware nor
AWS. Lifecycle mirrors DockerProvider: every start_emulator() boots a *clean*
guest (the base qcow2 is opened with snapshot=on, so it is never modified) and
revert_to_snapshot() just stops the VM, after which DesktopEnv.reset() calls
start_emulator() again.

Configuration (environment variables):
  OSWORLD_QEMU_BIN            qemu-system-x86_64 binary (default: on PATH)
  OSWORLD_QEMU_DISK           path to the extracted osworld-v2-ubuntu-x86.qcow2 (required)
  OSWORLD_QEMU_FIRMWARE_DIR   dir containing OVMF_CODE_4M.fd and OVMF_VARS.fd (required)
  OSWORLD_QEMU_RUN_DIR        scratch dir for pid/log/OVMF vars (default ./qemu_run)
  OSWORLD_QEMU_TMPDIR         where QEMU puts the throw-away snapshot overlay
                              (default $TMPDIR; NOTE /tmp is RAM-backed on many clusters)
  OSWORLD_QEMU_RAM / _SMP     guest memory / vCPUs (default 4G / 4)
  OSWORLD_QEMU_ACCEL          "kvm", "tcg" or "auto" (default auto: kvm if /dev/kvm is usable)
  OSWORLD_QEMU_{SERVER,CHROMIUM,VNC,VLC}_PORT
                              host ports (defaults 15000 / 19222 / 5900 / 18080); all are bound
                              to 127.0.0.1 only -- the guest agent offers remote command execution
  OSWORLD_QEMU_EXTRA_FORWARDS extra "host:guest,host:guest" loopback forwards
                              (default "3000:3000,8000:8000" -- OSWorld-V2 task services)
  OSWORLD_QEMU_BOOT_TIMEOUT   seconds to wait for the guest agent (default 900)
  OSWORLD_QEMU_SETTLE_SECONDS extra wait after the agent answers so the desktop can render
                              (default 90 under TCG, 20 under KVM)
  OSWORLD_QEMU_ATTACH=1       do not launch/stop anything; use an already-running VM

Only one VM at a time (fixed host ports): run with --num_envs 1.
"""

import logging
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests

from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.qemu.QemuProvider")
logger.setLevel(logging.INFO)


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


_DEFAULT_QEMU_BIN = "/usr/bin/qemu-system-x86_64"
_DEFAULT_FIRMWARE_DIR = "/lambda/nfs/psychological-steering/benchmarks/osworld/firmware"


class QemuProvider(Provider):
    def __init__(self, region: str = None):
        super().__init__(region)
        env = os.environ.get
        self.attach = env("OSWORLD_QEMU_ATTACH", "0") == "1"
        self.qemu_bin = env("OSWORLD_QEMU_BIN") or shutil.which("qemu-system-x86_64") or _DEFAULT_QEMU_BIN
        self.firmware_dir = env("OSWORLD_QEMU_FIRMWARE_DIR", _DEFAULT_FIRMWARE_DIR)
        self.run_dir = Path(env("OSWORLD_QEMU_RUN_DIR", "./qemu_run")).resolve()
        self.tmpdir = env("OSWORLD_QEMU_TMPDIR") or env("TMPDIR")
        self.ram = env("OSWORLD_QEMU_RAM", "4G")
        self.smp = env("OSWORLD_QEMU_SMP", "4")
        self.boot_timeout = int(env("OSWORLD_QEMU_BOOT_TIMEOUT", "900"))
        self.server_port = int(env("OSWORLD_QEMU_SERVER_PORT", "15000"))
        self.chromium_port = int(env("OSWORLD_QEMU_CHROMIUM_PORT", "19222"))
        self.vnc_port = int(env("OSWORLD_QEMU_VNC_PORT", "5900"))
        self.vlc_port = int(env("OSWORLD_QEMU_VLC_PORT", "18080"))
        self.extra_forwards = []
        for item in env("OSWORLD_QEMU_EXTRA_FORWARDS", "3000:3000,8000:8000").split(","):
            item = item.strip()
            if item:
                host, guest = item.split(":")
                self.extra_forwards.append((int(host), int(guest)))

        accel = env("OSWORLD_QEMU_ACCEL", "auto").lower()
        if accel == "auto":
            accel = "kvm" if os.access("/dev/kvm", os.R_OK | os.W_OK) else "tcg"
        self.accel = accel
        self.settle = int(env("OSWORLD_QEMU_SETTLE_SECONDS", "20" if accel == "kvm" else "90"))
        self.proc = None

    # -- helpers -----------------------------------------------------------------
    def _agent_ready(self) -> bool:
        try:
            r = requests.get(f"http://localhost:{self.server_port}/screenshot", timeout=(5, 10))
            return r.status_code == 200
        except Exception:
            return False

    def _build_cmd(self, disk: str, vars_copy: Path) -> list:
        if not self.firmware_dir:
            raise RuntimeError("OSWORLD_QEMU_FIRMWARE_DIR is not set (needs OVMF_CODE_4M.fd + OVMF_VARS.fd).")
        code_fd = Path(self.firmware_dir) / "OVMF_CODE_4M.fd"
        if self.accel == "kvm":
            machine, cpu = "q35,accel=kvm", "host"
        else:
            # `-cpu max` under TCG advertises AVX-512, which TCG cannot emulate; GNOME Shell then
            # crashes in llvmpipe and the screen stays black. qemu64 is safe.
            machine, cpu = "q35,accel=tcg", "qemu64"
        fwd = [
            f"hostfwd=tcp:127.0.0.1:{self.server_port}-:5000",
            f"hostfwd=tcp:127.0.0.1:{self.chromium_port}-:9222",
            f"hostfwd=tcp:127.0.0.1:{self.vlc_port}-:8080",
        ] + [f"hostfwd=tcp:127.0.0.1:{h}-:{g}" for h, g in self.extra_forwards]
        return [
            self.qemu_bin, "-name", "osworld-v2",
            "-machine", machine, "-cpu", cpu, "-smp", self.smp, "-m", self.ram,
            "-drive", f"if=pflash,format=raw,readonly=on,file={code_fd}",
            "-drive", f"if=pflash,format=raw,file={vars_copy}",
            "-drive", f"file={disk},if=none,id=hda,format=qcow2,cache=writeback,discard=unmap,snapshot=on",
            "-device", "virtio-blk-pci,drive=hda",
            "-netdev", "user,id=net0," + ",".join(fwd),
            "-device", "virtio-net-pci,netdev=net0",
            "-object", "rng-random,filename=/dev/urandom,id=rng0", "-device", "virtio-rng-pci,rng=rng0",
            "-vnc", f"127.0.0.1:{self.vnc_port - 5900}",
            "-display", "none",
            "-serial", f"file:{self.run_dir / 'serial.log'}",
        ]

    # -- Provider interface --------------------------------------------------------
    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"):
        if self.attach:
            logger.info("QEMU provider in attach mode: using the already-running VM.")
            self._wait_ready()
            return
        if self.proc is not None and self.proc.poll() is None:
            self.stop_emulator(path_to_vm)
        if not Path(path_to_vm).exists():
            raise FileNotFoundError(f"VM disk not found: {path_to_vm} (set OSWORLD_QEMU_DISK)")
        ports = [self.server_port, self.chromium_port, self.vlc_port] + [h for h, _ in self.extra_forwards]
        busy = [p for p in ports if not _port_free(p)]
        if busy:
            raise RuntimeError(f"Host ports already in use: {busy}. Stop the other VM or set OSWORLD_QEMU_*_PORT.")

        self.run_dir.mkdir(parents=True, exist_ok=True)
        vars_copy = self.run_dir / "OVMF_VARS.fd"
        shutil.copyfile(Path(self.firmware_dir or "") / "OVMF_VARS.fd", vars_copy)

        cmd = self._build_cmd(path_to_vm, vars_copy)
        env = dict(os.environ)
        if self.tmpdir:
            Path(self.tmpdir).mkdir(parents=True, exist_ok=True)
            env["TMPDIR"] = self.tmpdir
        log = open(self.run_dir / "qemu_stdout.log", "wb")
        logger.info("Starting QEMU (%s, %s vCPU, %s RAM): %s", self.accel, self.smp, self.ram, " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        (self.run_dir / "qemu.pid").write_text(str(self.proc.pid))
        self._wait_ready()

    def _wait_ready(self):
        start = time.time()
        while time.time() - start < self.boot_timeout:
            if self.proc is not None and self.proc.poll() is not None:
                tail = (self.run_dir / "qemu_stdout.log").read_text(errors="replace")[-2000:]
                raise RuntimeError(f"QEMU exited early (code {self.proc.returncode}):\n{tail}")
            if self._agent_ready():
                logger.info("Guest agent is up after %.0fs; waiting %ss for the desktop to settle.",
                            time.time() - start, self.settle)
                time.sleep(self.settle)
                return
            time.sleep(5)
        raise TimeoutError(f"Guest agent did not answer on port {self.server_port} within {self.boot_timeout}s")

    def get_ip_address(self, path_to_vm: str) -> str:
        return f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for the QEMU provider (each start is a clean boot).")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        # DesktopEnv.reset() calls start_emulator() next, which boots a fresh clean guest.
        self.stop_emulator(path_to_vm)
        return path_to_vm

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        if self.attach:
            logger.info("QEMU provider in attach mode: leaving the VM running.")
            return
        if self.proc is None:
            return
        logger.info("Stopping VM (pid %s)...", self.proc.pid)
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=15)
        except ProcessLookupError:
            pass
        finally:
            self.proc = None
