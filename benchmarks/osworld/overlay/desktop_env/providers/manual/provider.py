import logging
import os

from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.manual.ManualProvider")

# This cluster has no Docker/KVM/AWS access, so the VM is launched by hand via
# a from-source QEMU build (see osworld/vm_build/) instead of through any of
# OSWorld's normal provider lifecycles. This provider just points DesktopEnv
# at that already-running instance's forwarded ports rather than managing a
# VM itself.
MANUAL_SERVER_PORT = int(os.environ.get("OSWORLD_MANUAL_SERVER_PORT", "15000"))
MANUAL_CHROMIUM_PORT = int(os.environ.get("OSWORLD_MANUAL_CHROMIUM_PORT", "19222"))
MANUAL_VNC_PORT = int(os.environ.get("OSWORLD_MANUAL_VNC_PORT", "5900"))
MANUAL_VLC_PORT = int(os.environ.get("OSWORLD_MANUAL_VLC_PORT", "18080"))


class ManualProvider(Provider):
    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"):
        logger.info("Manual provider: assuming the VM is already running (see osworld/vm_build/run/).")

    def get_ip_address(self, path_to_vm: str) -> str:
        return f"localhost:{MANUAL_SERVER_PORT}:{MANUAL_CHROMIUM_PORT}:{MANUAL_VNC_PORT}:{MANUAL_VLC_PORT}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for the manual provider.")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        logger.warning("Manual provider: no snapshot revert -- reusing the VM's current state as-is.")
        return path_to_vm

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        logger.info("Manual provider: leaving the VM running (stop it yourself via osworld/vm_build/run/qemu.pid).")
