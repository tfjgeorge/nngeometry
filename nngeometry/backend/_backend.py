from torch.utils.data import DataLoader, TensorDataset
import torch
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

    def _infer_differentiable_leafs(self, input, mods):
        if input.dtype in differentiable_dtypes:
            input.requires_grad = True
            return [input]
        else:
            # find Embeddings
            embedding_parameters = []
            for module in mods:
                if isinstance(module, torch.nn.Embedding):
                    embedding_parameters.append(next(module.parameters()))
            if len(embedding_parameters) > 0:
                return embedding_parameters

        # Otherwise return all differentiable params
        params = []
        for module in mods:
            params.append(next(module.parameters()))
        return params

    def _get_iter_loader(self, loader):
        if self.verbose:
            return tqdm(iter(loader))
        else:
            return iter(loader)
