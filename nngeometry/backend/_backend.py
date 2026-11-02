import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

differentiable_dtypes = [torch.float16, torch.float32, torch.float64]


class AbstractBackend:
    def _check_same_device(self, mods):
        device = None
        for mod in mods:
            if device is None:
                device = self._infer_device(mod)
            elif device != self._infer_device(mod):
                raise ValueError("All modules should be on the same device")
        return device

    def _check_same_dtype(self, mods):
        dtype = None
        for mod in mods:
            if dtype is None:
                dtype = self._infer_dtype(mod)
            elif dtype != self._infer_dtype(mod):
                raise ValueError("All modules should have the same type")
        return dtype

    def get_device(self, layer_collection):
        return self._check_same_device(layer_collection)

    def get_dtype(self, layer_collection):
        return self._check_same_dtype(layer_collection)

    def _get_dataloader(self, examples):
        if isinstance(examples, DataLoader):
            return examples
        else:
            return DataLoader(TensorDataset(*examples), batch_size=len(examples[0]))

    def _infer_dtype(self, mod):
        return mod.weight.dtype

    def _infer_device(self, mod):
        return mod.weight.device

    def _get_iter_loader(self, loader):
        if self.verbose:
            return tqdm(iter(loader))
        else:
            return iter(loader)
