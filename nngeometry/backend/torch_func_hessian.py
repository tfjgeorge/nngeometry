from functools import partial

import torch

from nngeometry.object.map import PFMapDense
from nngeometry.object.vector import PVector

from ._backend import AbstractBackend


def hvp(func, primals, tangents):
    return torch.func.jvp(
        lambda p: torch.func.grad(func)(p),
        primals=(primals,),
        tangents=(tangents,),
    )[1]


def batched_hvp(func, primals, batched_tangents):
    return torch.vmap(lambda tangents: hvp(func, primals, tangents))(batched_tangents)


class TorchFuncHessianBackend(AbstractBackend):
    def __init__(self, model, function, verbose=False):
        self.model = model
        self.function = function
        self.verbose = verbose

    def get_covariance_matrix(self, examples, layer_collection):
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())

        loader = self._get_dataloader(examples)
        n_parameters = layer_collection.numel()
        H = torch.zeros((n_parameters, n_parameters), device=device, dtype=dtype)

        def compute_loss(params, inputs, targets):
            prediction = torch.func.functional_call(self.model, params, (inputs,))
            return self.function(prediction, targets)

        params_dict = dict(layer_collection.named_parameters(layerid_to_mod))
        params_dict = {k: v.detach() for k, v in params_dict.items()}

        for d in self._get_iter_loader(loader):
            inputs = d[0].to(device)
            targets = d[1].to(device)

            H_mb = torch.func.hessian(
                partial(compute_loss, inputs=inputs, targets=targets),
            )(params_dict)

            for layer_id_x, layer_x in layer_collection.layers.items():
                start_x = layer_collection.p_pos[layer_id_x]
                for layer_id_y, layer_y in layer_collection.layers.items():
                    ws_x = layer_x.weight.numel()
                    ws_y = layer_y.weight.numel()
                    start_y = layer_collection.p_pos[layer_id_y]

                    # weight_x, weight_y
                    H[
                        start_x : start_x + ws_x,
                        start_y : start_y + ws_y,
                    ] += (
                        H_mb[layer_id_x + ".weight"][layer_id_y + ".weight"]
                        .reshape(ws_x, ws_y)
                        .detach()
                    )

                    if layer_x.has_bias():
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

                        if layer_y.has_bias():
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

                    if layer_y.has_bias():
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

    def implicit_mv(self, v, examples, layer_collection):
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        device = self._check_same_device(layerid_to_mod.values())

        loader = self._get_dataloader(examples)

        def compute_loss(params, inputs, targets):
            prediction = torch.func.functional_call(self.model, params, (inputs,))
            return self.function(prediction, targets)

        params_dict = dict(layer_collection.named_parameters(layerid_to_mod))
        params_dict = {k: v.detach() for k, v in params_dict.items()}

        v_dict = {}  # replace with function in PVector ?
        for key, value in v.to_dict().items():
            if len(value) > 1:
                v_dict[key + ".weight"] = value[0]
                v_dict[key + ".bias"] = value[1]
            else:
                v_dict[key + ".weight"] = value[0]

        hvp_dict = {k: torch.zeros_like(p) for k, p in params_dict.items()}

        for d in self._get_iter_loader(loader):
            inputs = d[0].to(device)
            targets = d[1].to(device)

            hvp_mb = hvp(
                partial(compute_loss, inputs=inputs, targets=targets),
                params_dict,
                v_dict,
            )

            for k in hvp_mb:
                hvp_dict[k] += hvp_mb[k].detach()

        output_dict = dict()
        for layer_id, layer in layer_collection.layers.items():
            if layer.has_bias():
                output_dict[layer_id] = (
                    hvp_dict[layer_id + ".weight"],
                    hvp_dict[layer_id + ".bias"],
                )
            else:
                output_dict[layer_id] = (hvp_dict[layer_id + ".weight"],)

        return PVector(layer_collection, dict_repr=output_dict)

    def implicit_mmap(self, pfmap, examples, layer_collection):
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        device = self._check_same_device(layerid_to_mod.values())

        loader = self._get_dataloader(examples)

        def compute_loss(params, inputs, targets):
            prediction = torch.func.functional_call(self.model, params, (inputs,))
            return self.function(prediction, targets)

        so, sb, *_ = pfmap.size()

        params_dict = dict(layer_collection.named_parameters(layerid_to_mod))
        params_dict = {k: v.detach() for k, v in params_dict.items()}

        pfmap_dict = {}
        for layer_id, layer in layer_collection.layers.items():
            d = pfmap.to_torch_layer(layer_id)
            if layer.has_bias():
                pfmap_dict[layer_id + ".weight"] = d[0].view(-1, *layer.weight.size)
                pfmap_dict[layer_id + ".bias"] = d[1].view(-1, *layer.bias.size)
            else:
                pfmap_dict[layer_id + ".weight"] = d[0].view(-1, *layer.weight.size)

        b_hvp_dict = {
            k: torch.zeros((so * sb, *p.shape), dtype=p.dtype, device=p.device)
            for k, p in params_dict.items()
        }

        for d in self._get_iter_loader(loader):
            inputs = d[0].to(device)
            targets = d[1].to(device)

            b_hvp_mb = batched_hvp(
                partial(compute_loss, inputs=inputs, targets=targets),
                params_dict,
                pfmap_dict,
            )

            for k in b_hvp_mb:
                b_hvp_dict[k] += b_hvp_mb[k].detach()

        output_dict = dict()
        for layer_id, layer in layer_collection.layers.items():
            if layer.has_bias():
                output_dict[layer_id] = (
                    b_hvp_dict[layer_id + ".weight"].view(so, sb, -1),
                    b_hvp_dict[layer_id + ".bias"].view(so, sb, -1),
                )
            else:
                output_dict[layer_id] = (
                    b_hvp_dict[layer_id + ".weight"].view(so, sb, -1),
                )

        return PFMapDense.from_dict(
            generator=None, data_dict=output_dict, layer_collection=layer_collection
        )
