import os

from desktop_env.providers.base import VMManager

# The VM disk is a single pre-extracted qcow2 whose path comes from
# OSWORLD_QEMU_DISK; nothing here tracks a real VM inventory. Defaults to the
# V1 Ubuntu.qcow2 already downloaded+boot-tested for this tree (vm_build/),
# same image WeaveBench's own "docker" provider pulls from xlangai/ubuntu_osworld.


class QemuVMManager(VMManager):
    def initialize_registry(self, **kwargs):
        pass

    def add_vm(self, vm_path, **kwargs):
        pass

    def delete_vm(self, vm_path, **kwargs):
        pass

    def occupy_vm(self, vm_path, pid, **kwargs):
        pass

    def list_free_vms(self, **kwargs):
        return [self.get_vm_path()]

    def check_and_clean(self, **kwargs):
        pass

    def get_vm_path(self, **kwargs):
        disk = os.environ.get(
            "OSWORLD_QEMU_DISK",
            "/project2/emiliofe_74/prabhu/osworld/vm_build/vm_data/Ubuntu.qcow2",
        )
        return os.path.abspath(os.path.expanduser(disk))
