from torch.utils.data import DataLoader, TensorDataset
import torch

differentiable_dtypes = [torch.float16, torch.float32, torch.float64]


class AbstractBackend:

    def _check_same_device(self):
        device = None
        for layer_id in self.layer_collection.layers.keys():
            if device is None:
                device = self._infer_device(layer_id)
            elif device != self._infer_device(layer_id):
                raise ValueError("All modules should reside on the same device")
        return device

    def _check_same_dtype(self):
        dtype = None
        for layer_id in self.layer_collection.layers.keys():
            if dtype is None:
                dtype = self._infer_dtype(layer_id)
            elif dtype != self._infer_dtype(layer_id):
                raise ValueError("All modules should have the same type")
        return dtype

    def get_device(self):
        return self._check_same_device()

    def get_dtype(self):
        return self._check_same_dtype()

    def _get_dataloader(self, examples):
        if isinstance(examples, DataLoader):
            return examples
        else:
            return DataLoader(TensorDataset(*examples), batch_size=len(examples[0]))

    def _infer_dtype(self, layer_id):
        return self.l_to_m[layer_id].weight.dtype

    def _infer_device(self, layer_id):
        return self.l_to_m[layer_id].weight.device

    def _infer_differentiable_leafs(self, input):
        if input.dtype in differentiable_dtypes:
            input.requires_grad = True
            return [input]
        else:
            # find Embeddings
            embedding_parameters = []
            for module in self.l_to_m.values():
                if isinstance(module, torch.nn.Embedding):
                    embedding_parameters.append(next(module.parameters()))
            return embedding_parameters
