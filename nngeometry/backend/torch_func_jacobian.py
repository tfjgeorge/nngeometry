from functools import partial

import torch

from nngeometry.object.map import PFMapDense
from nngeometry.object.vector import PVector

from ._backend import AbstractBackend


def fvp(func, primals, tangents):
    _, jvp_out = torch.func.jvp(
        lambda p: func(p),
        primals=(primals,),
        tangents=(tangents,),
    )
    _, vjp_fn = torch.func.vjp(func, primals)
    return vjp_fn(jvp_out)[0]


def batched_fvp(func, primals, batched_tangents):
    return torch.vmap(lambda tangents: fvp(func, primals, tangents))(batched_tangents)


class TorchFuncJacobianBackend(AbstractBackend):
    def __init__(self, model, function, verbose=False):
        self.model = model
        self.function = function
        self.verbose = verbose

    def implicit_mv(self, v, examples, layer_collection):
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        device = self._check_same_device(layerid_to_mod.values())

        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)

        def function(params, inputs, targets=None):
            predictions = torch.func.functional_call(self.model, params, (inputs,))
            if targets is None:
                return self.function(predictions)
            else:
                return self.function(predictions, targets)

        params_dict = dict(layer_collection.named_parameters(layerid_to_mod))
        params_dict = {k: v.detach() for k, v in params_dict.items()}

        v_dict = {}  # replace with function in PVector ?
        for key, value in v.to_dict().items():
            if len(value) > 1:
                v_dict[key + ".weight"] = value[0]
                v_dict[key + ".bias"] = value[1]
            else:
                v_dict[key + ".weight"] = value[0]

        fvp_dict = {k: torch.zeros_like(p) for k, p in params_dict.items()}

        for d in self._get_iter_loader(loader):
            inputs = d[0].to(device)
            if len(d) > 1:
                targets = d[1].to(device)
            else:
                targets = None

            fvp_mb = fvp(
                partial(function, inputs=inputs, targets=targets), params_dict, v_dict
            )

            for k in fvp_mb:
                fvp_dict[k] += fvp_mb[k].detach()

        for k in fvp_dict:
            fvp_dict[k] /= n_examples

        output_dict = dict()
        for layer_id, layer in layer_collection.layers.items():
            if layer.has_bias():
                output_dict[layer_id] = (
                    fvp_dict[layer_id + ".weight"],
                    fvp_dict[layer_id + ".bias"],
                )
            else:
                output_dict[layer_id] = (fvp_dict[layer_id + ".weight"],)

        return PVector(layer_collection, dict_repr=output_dict)

    def implicit_mmap(self, pfmap, examples, layer_collection):
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        device = self._check_same_device(layerid_to_mod.values())

        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)

        def function(params, inputs, targets):
            predictions = torch.func.functional_call(self.model, params, (inputs,))
            if targets is None:
                return self.function(predictions)
            else:
                return self.function(predictions, targets)

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

        b_fvp_dict = {
            k: torch.zeros((so * sb, *p.shape), dtype=p.dtype, device=p.device)
            for k, p in params_dict.items()
        }

        for d in self._get_iter_loader(loader):
            inputs = d[0].to(device)
            if len(d) > 1:
                targets = d[1].to(device)
            else:
                targets = None

            b_fvp_mb = batched_fvp(
                partial(function, inputs=inputs, targets=targets),
                params_dict,
                pfmap_dict,
            )

            for k in b_fvp_mb:
                b_fvp_dict[k] += b_fvp_mb[k].detach()

        for k in b_fvp_dict:
            b_fvp_dict[k] /= n_examples

        output_dict = dict()
        for layer_id, layer in layer_collection.layers.items():
            if layer.has_bias():
                output_dict[layer_id] = (
                    b_fvp_dict[layer_id + ".weight"].view(so, sb, -1),
                    b_fvp_dict[layer_id + ".bias"].view(so, sb, -1),
                )
            else:
                output_dict[layer_id] = (
                    b_fvp_dict[layer_id + ".weight"].view(so, sb, -1),
                )

        return PFMapDense.from_dict(
            generator=None, data_dict=output_dict, layer_collection=layer_collection
        )
