from desktop_env.providers.base import VMManager

# See provider.py -- this manager doesn't track any real VM inventory, it
# just satisfies DesktopEnv's interface for the single hand-launched VM.


class ManualVMManager(VMManager):
    def initialize_registry(self, **kwargs):
        pass

    def add_vm(self, vm_path, **kwargs):
        pass

    def delete_vm(self, vm_path, **kwargs):
        pass

    def occupy_vm(self, vm_path, pid, **kwargs):
        pass

    def list_free_vms(self, **kwargs):
        return ["manual-vm"]

    def check_and_clean(self, **kwargs):
        pass

    def get_vm_path(self, os_type="Ubuntu", region=None, screen_size=(1920, 1080), **kwargs):
        return "manual-vm"
