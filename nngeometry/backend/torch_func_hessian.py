import torch
from nngeometry.layercollection import LayerCollection
from ._backend import AbstractBackend


class TorchFuncHessianBackend(AbstractBackend):

    def __init__(self, model, function, layer_collection=None):
        self.model = model
        self.function = function

        if layer_collection is None:
            self.layer_collection = LayerCollection.from_model(model)
        else:
            self.layer_collection = layer_collection

        # maps parameters to their position in flattened representation
        self.l_to_m, self.m_to_l = self.layer_collection.get_layerid_module_maps(model)

        self.params_dict = dict(self.layer_collection.named_parameters(self.l_to_m))

    def get_covariance_matrix(self, examples):
        device = self._check_same_device()
        dtype = self._check_same_dtype()

        loader = self._get_dataloader(examples)
        n_parameters = self.layer_collection.numel()
        H = torch.zeros((n_parameters, n_parameters), device=device, dtype=dtype)

        def compute_loss(params, X, y):
            prediction = torch.func.functional_call(self.model, params, (X,))
            return self.function(prediction, y)

        for d in loader:
            inputs = d[0]

            H_mb = torch.func.hessian(compute_loss)(self.params_dict, inputs, d[1])

            for layer_id_x, layer_x in self.layer_collection.layers.items():
                start_x = self.layer_collection.p_pos[layer_id_x]
                for layer_id_y, layer_y in self.layer_collection.layers.items():
                    ws_x = layer_x.weight.numel()
                    ws_y = layer_y.weight.numel()
                    start_y = self.layer_collection.p_pos[layer_id_y]

                    # weight_x, weight_y
                    H[
                        start_x : start_x + ws_x,
                        start_y : start_y + ws_y,
                    ] += (
                        H_mb[layer_id_x + ".weight"][layer_id_y + ".weight"]
                        .reshape(ws_x, ws_y)
                        .detach()
                    )

                    if layer_x.bias is not None:
                        bs_x = layer_x.bias.numel()

                        # bias_x, weight_y
                        H[
                            start_x + ws_x : start_x + ws_x + bs_x,
                            start_y : start_y + ws_y,
                        ] += (
                            H_mb[layer_id_x + ".bias"][layer_id_y + ".weight"]
                            .reshape(bs_x, ws_y)
                            .detach()
                        )

                        if layer_y.bias is not None:
                            bs_y = layer_y.bias.numel()

                            # bias_x, bias_y
                            H[
                                start_x + ws_x : start_x + ws_x + bs_x,
                                start_y + ws_y : start_y + ws_y + bs_y,
                            ] += (
                                H_mb[layer_id_x + ".bias"][layer_id_y + ".bias"]
                                .reshape(bs_x, bs_y)
                                .detach()
                            )

                    if layer_y.bias is not None:
                        bs_y = layer_y.bias.numel()

                        # weight_x, bias_y
                        H[
                            start_x : start_x + ws_x,
                            start_y + ws_y : start_y + ws_y + bs_y,
                        ] += (
                            H_mb[layer_id_x + ".weight"][layer_id_y + ".bias"]
                            .reshape(ws_x, bs_y)
                            .detach()
                        )

        return H
