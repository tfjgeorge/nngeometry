from functools import partial

import torch

from nngeometry.layercollection import LayerCollection
from nngeometry.object.vector import FVector, PVector

from .._backend import AbstractBackend
from .grads import FactoryMap


class TorchHooksJacobianBackend(AbstractBackend):
    """
    Computes jacobians
    :math:`\mathbf{J}_{ijk}=\\frac{\partial f\left(x_{j}\\right)_{i}}{\delta\mathbf{w}_{k}}`,
    FIM matrices :math:`\mathbf{F}_{k,k'}=\\frac{1}{n}\sum_{i,j}\mathbf{J}_{ijk}\mathbf{J}_{ijk'}`
    and NTK matrices :math:`\mathbf{K}_{iji'j'}=\sum_{k}\mathbf{J}_{ijk}\mathbf{J}_{ijk'}`.

    This generator is written in pure PyTorch and exploits some tricks in order to make
    computations more efficient.

    :param layer_collection:
    :type layer_collection: :class:`.layercollection.LayerCollection`
    :param model:
    :type model: Pytorch `nn.Module`
    :param function: A function :math:`f\left(X,Y,Z\\right)` where :math:`X,Y,Z` are
        minibatchs returned by the dataloader (Note that in some cases :math:`Y,Z` are
        not required). If None, it defaults to `function = lambda *x: model(x[0])`
    :type function: python function

    """

    def __init__(self, model, function=None, centering=False):
        self.model = model
        self.handles = []
        self.centering = centering

        # this contains functions that require knowledge of number of
        # outputs, not known before first minibatch
        self.delayed_for_n_ouput = []

        if function is None:

            def function(*x):
                return model(x[0])

        self.function = function

    def get_covariance_matrix(self, examples, layer_collection):
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        # add hooks
        self.handles += self._add_hooks(
            self._hook_savex,
            self._hook_compute_flat_grad,
            layerid_to_mod,
            layer_collection,
        )

        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        n_parameters = layer_collection.numel()
        bs = loader.batch_size
        G = torch.zeros((n_parameters, n_parameters), device=device, dtype=dtype)
        self.grads = torch.zeros((1, bs, n_parameters), device=device, dtype=dtype)
        if self.centering:

            def f(n_output):
                self._grad_mean = torch.zeros(
                    (n_output, n_parameters), device=device, dtype=dtype
                )

            self.delayed_for_n_ouput.append(f)

        self.start = 0
        self.i_output = 0
        for d in loader:
            self.xs = dict()
            inputs = d[0]
            grad_wrt = self._infer_differentiable_leafs(inputs, layerid_to_mod.values())
            bs = inputs.size(0)
            output = self.function(*d).view(bs, -1).sum(dim=0)
            n_output = output.size(-1)
            self._exec_delayed_n_output(n_output)
            for i in range(n_output):
                self.grads.zero_()
                torch.autograd.grad(
                    output[i],
                    grad_wrt,
                    retain_graph=i < n_output - 1,
                    only_inputs=True,
                )
                G += torch.mm(self.grads[0].t(), self.grads[0])
                if self.centering:
                    self._grad_mean[i].add_(self.grads[0].sum(dim=0))
        G /= n_examples
        if self.centering:
            self._grad_mean /= n_examples
            G -= torch.mm(self._grad_mean.t(), self._grad_mean)

        # remove hooks
        del self.grads
        del self.xs
        for h in self.handles:
            h.remove()

        if self.centering:
            del self._grad_mean

        return G

    def get_covariance_diag(self, examples, layer_collection):
        if self.centering:
            raise NotImplementedError
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        # add hooks
        self.handles += self._add_hooks(
            self._hook_savex,
            self._hook_compute_diag,
            layerid_to_mod,
            layer_collection,
        )

        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        n_parameters = layer_collection.numel()
        self.diag_m = torch.zeros((n_parameters,), device=device, dtype=dtype)
        self.start = 0
        for d in loader:
            self.xs = dict()
            inputs = d[0]
            grad_wrt = self._infer_differentiable_leafs(inputs, layerid_to_mod.values())

            bs = inputs.size(0)
            output = self.function(*d).view(bs, -1).sum(dim=0)
            n_output = output.size(-1)

            for i in range(n_output):
                torch.autograd.grad(
                    output[i],
                    grad_wrt,
                    retain_graph=i < n_output - 1,
                    only_inputs=True,
                )
        diag_m = self.diag_m / n_examples

        # remove hooks
        del self.diag_m
        del self.xs
        for h in self.handles:
            h.remove()

        return diag_m

    def get_covariance_quasidiag(self, examples, layer_collection):
        if self.centering:
            raise NotImplementedError

        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        # add hooks
        self.handles += self._add_hooks(
            self._hook_savex,
            self._hook_compute_quasidiag,
            layerid_to_mod,
            layer_collection,
        )
        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())

        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        self._blocks = dict()
        for layer_id, layer in layer_collection.layers.items():
            s = layer.numel()
            if layer.bias is None:
                self._blocks[layer_id] = (
                    torch.zeros((s,), device=device, dtype=dtype),
                    None,
                )
            else:
                cross_s = layer.weight.size
                self._blocks[layer_id] = (
                    torch.zeros((s,), device=device, dtype=dtype),
                    torch.zeros(cross_s, device=device, dtype=dtype),
                )

        for d in loader:
            self.xs = dict()
            inputs = d[0]
            grad_wrt = self._infer_differentiable_leafs(inputs, layerid_to_mod.values())
            bs = inputs.size(0)
            output = self.function(*d).view(bs, -1).sum(dim=0)
            n_output = output.size(-1)
            for i in range(n_output):
                torch.autograd.grad(
                    output[i],
                    grad_wrt,
                    retain_graph=i < n_output - 1,
                    only_inputs=True,
                )
        for d, c in self._blocks.values():
            d.div_(n_examples)
            if c is not None:
                c.div_(n_examples)

        blocks = self._blocks

        # remove hooks
        del self._blocks
        del self.xs
        for h in self.handles:
            h.remove()

        return blocks

    def get_covariance_layer_blocks(self, examples, layer_collection):
        if self.centering:
            raise NotImplementedError
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        # add hooks
        self.handles += self._add_hooks(
            self._hook_savex,
            self._hook_compute_layer_blocks,
            layerid_to_mod,
            layer_collection,
        )

        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())

        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        self._blocks = dict()
        for layer_id, layer in layer_collection.layers.items():
            s = layer.numel()
            self._blocks[layer_id] = torch.zeros((s, s), device=device, dtype=dtype)

        for d in loader:
            self.xs = dict()
            inputs = d[0]
            grad_wrt = self._infer_differentiable_leafs(inputs, layerid_to_mod.values())

            bs = inputs.size(0)
            output = self.function(*d).view(bs, -1).sum(dim=0)
            n_output = output.size(-1)
            for i in range(n_output):
                torch.autograd.grad(
                    output[i],
                    grad_wrt,
                    retain_graph=i < n_output - 1,
                    only_inputs=True,
                )
        blocks = {m: self._blocks[m] / n_examples for m in self._blocks.keys()}

        # remove hooks
        del self._blocks
        del self.xs
        for h in self.handles:
            h.remove()

        return blocks

    def get_kfac_blocks(self, examples, layer_collection):
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        # add hooks
        self.handles += self._add_hooks(
            self._hook_savex,
            self._hook_compute_kfac_blocks,
            layerid_to_mod,
            layer_collection,
        )

        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())

        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        self._blocks = dict()
        for layer_id, layer in layer_collection.layers.items():
            layer_class = layer.__class__.__name__
            if layer_class == "LinearLayer":
                sG = layer.out_features
                sA = layer.in_features
            elif layer_class == "Conv2dLayer":
                sG = layer.out_channels
                sA = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            elif layer_class == "Conv1dLayer":
                sG = layer.out_channels
                sA = layer.in_channels * layer.kernel_size[0]
            elif layer_class == "EmbeddingLayer":
                sG = layer.embedding_dim
                sA = layer.num_embeddings
            if layer.has_bias():
                sA += 1
            self._blocks[layer_id] = (
                torch.zeros((sA, sA), device=device, dtype=dtype),
                torch.zeros((sG, sG), device=device, dtype=dtype),
            )

        for d in loader:
            self.xs = dict()
            inputs = d[0]
            grad_wrt = self._infer_differentiable_leafs(inputs, layerid_to_mod.values())

            bs = inputs.size(0)
            output = self.function(*d).view(bs, -1).sum(dim=0)
            n_output = output.size(-1)
            for self.i_output in range(n_output):
                retain_graph = self.i_output < n_output - 1
                torch.autograd.grad(
                    output[self.i_output],
                    grad_wrt,
                    retain_graph=retain_graph,
                    only_inputs=True,
                )
        for layer_id in layer_collection.layers.keys():
            self._blocks[layer_id][0].div_(n_examples / n_output**0.5)
            self._blocks[layer_id][1].div_(n_output**0.5 * n_examples)
        blocks = self._blocks

        # remove hooks
        del self._blocks
        del self.i_output
        del self.xs
        for h in self.handles:
            h.remove()

        return blocks

    def get_jacobian(self, examples, layer_collection):
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        # add hooks
        self.handles += self._add_hooks(
            self._hook_savex,
            self._hook_compute_flat_grad,
            layerid_to_mod,
            layer_collection,
        )

        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        n_parameters = layer_collection.numel()

        def _f(n_output):
            self.grads = torch.zeros(
                (n_output, n_examples, n_parameters), device=device, dtype=dtype
            )

        self.delayed_for_n_ouput.append(_f)

        self.start = 0
        for d in loader:
            self.xs = dict()
            inputs = d[0]
            grad_wrt = self._infer_differentiable_leafs(inputs, layerid_to_mod.values())
            bs = inputs.size(0)
            output = self.function(*d).view(bs, -1).sum(dim=0)
            n_output = output.size(-1)
            self._exec_delayed_n_output(n_output)
            for self.i_output in range(n_output):
                retain_graph = self.i_output < n_output - 1
                torch.autograd.grad(
                    output[self.i_output],
                    grad_wrt,
                    retain_graph=retain_graph,
                    only_inputs=True,
                )
            self.start += inputs.size(0)
        grads = self.grads
        if self.centering:
            grads -= grads.mean(dim=1, keepdim=True)

        # remove hooks
        del self.grads
        del self.start
        del self.i_output
        del self.xs
        for h in self.handles:
            h.remove()

        return grads

    def get_gram_matrix(self, examples, layer_collection):
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        # add hooks
        self.handles += self._add_hooks(
            self._hook_savex_io,
            self._hook_kxy,
            layerid_to_mod,
            layer_collection,
        )

        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)

        def _f(n_output):
            self.G = torch.zeros(
                (n_output, n_examples, n_output, n_examples),
                device=device,
                dtype=dtype,
            )

        self.delayed_for_n_ouput.append(_f)

        self.gy_outer = dict()
        self.e_outer = 0
        for i_outer, d in enumerate(loader):
            self.x_outer = dict()
            # used in hooks to switch between store/compute
            inputs_outer = d[0]
            grad_wrt_outer = self._infer_differentiable_leafs(
                inputs_outer, layerid_to_mod.values()
            )
            bs_outer = inputs_outer.size(0)
            self.outerloop_switch = True
            output_outer = self.function(*d).view(bs_outer, -1).sum(dim=0)
            n_output = output_outer.size(-1)
            self._exec_delayed_n_output(n_output)

            for self.i_output_outer in range(n_output):
                self.outerloop_switch = True
                torch.autograd.grad(
                    output_outer[self.i_output_outer],
                    grad_wrt_outer,
                    retain_graph=True,
                    only_inputs=True,
                )
                self.outerloop_switch = False

                self.e_inner = 0
                for i_inner, d in enumerate(loader):
                    self.x_inner = dict()
                    if i_inner > i_outer:
                        break
                    inputs_inner = d[0]
                    grad_wrt_inner = self._infer_differentiable_leafs(
                        inputs_inner, layerid_to_mod.values()
                    )
                    bs_inner = inputs_inner.size(0)
                    output_inner = self.function(*d).view(bs_inner, n_output).sum(dim=0)
                    for self.i_output_inner in range(n_output):
                        torch.autograd.grad(
                            output_inner[self.i_output_inner],
                            grad_wrt_inner,
                            retain_graph=True,
                            only_inputs=True,
                        )

                    # since self.G is a symmetric matrix we only need to
                    # compute the upper or lower triangle
                    # => copy block and exclude diagonal
                    if i_inner < i_outer and self.i_output_outer == n_output - 1:
                        self.G[
                            :,
                            self.e_outer : self.e_outer + bs_outer,
                            :,
                            self.e_inner : self.e_inner + bs_inner,
                        ] += self.G[
                            :,
                            self.e_inner : self.e_inner + bs_inner,
                            :,
                            self.e_outer : self.e_outer + bs_outer,
                        ].permute(
                            2, 3, 0, 1
                        )
                    self.e_inner += inputs_inner.size(0)

            self.e_outer += inputs_outer.size(0)
        G = self.G
        if self.centering:
            C = (
                torch.eye(n_examples, device=G.device)
                - torch.ones((n_examples, n_examples), device=G.device) / n_examples
            )
            sG = G.size()
            G = torch.mm(G.view(-1, n_examples), C)
            G = (
                torch.mm(
                    C,
                    G.view(sG[0], sG[1], -1)
                    .permute(1, 0, 2)
                    .contiguous()
                    .view(n_examples, -1),
                )
                .view(sG[1], sG[0], -1)
                .permute(1, 0, 2)
                .contiguous()
                .view(*sG)
            )

        # remove hooks
        del self.e_inner, self.e_outer
        del self.G
        del self.x_inner
        del self.x_outer
        del self.gy_outer
        for h in self.handles:
            h.remove()

        return G

    def get_kfe_diag(self, kfe, examples, layer_collection):
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        # add hooks
        self.handles += self._add_hooks(
            self._hook_savex,
            self._hook_compute_kfe_diag,
            layerid_to_mod,
            layer_collection,
        )

        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())

        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        self._diags = dict()
        self._kfe = kfe
        for layer_id, layer in layer_collection.layers.items():
            layer_class = layer.__class__.__name__
            if layer_class == "LinearLayer":
                sG = layer.out_features
                sA = layer.in_features
            elif layer_class == "Conv2dLayer":
                sG = layer.out_channels
                sA = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            elif layer_class == "Conv1dLayer":
                sG = layer.out_channels
                sA = layer.in_channels * layer.kernel_size[0]
            elif layer_class == "EmbeddingLayer":
                sG = layer.embedding_dim
                sA = layer.num_embeddings
            if layer.has_bias():
                sA += 1
            self._diags[layer_id] = torch.zeros((sG * sA), device=device, dtype=dtype)

        for d in loader:
            self.xs = dict()
            inputs = d[0]
            grad_wrt = self._infer_differentiable_leafs(inputs, layerid_to_mod.values())
            bs = inputs.size(0)
            output = self.function(*d).view(bs, -1).sum(dim=0)
            n_output = output.size(-1)
            for self.i_output in range(n_output):
                retain_graph = self.i_output < n_output - 1
                torch.autograd.grad(
                    output[self.i_output],
                    grad_wrt,
                    retain_graph=retain_graph,
                    only_inputs=True,
                )
        diags = {
            l_id: self._diags[l_id] / n_examples
            for l_id in layer_collection.layers.keys()
        }

        # remove hooks
        del self._diags
        del self._kfe
        del self.xs
        for h in self.handles:
            h.remove()

        return diags

    def implicit_mv(self, v, examples, layer_collection):
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        # add hooks
        self.handles += self._add_hooks(
            self._hook_savex,
            self._hook_compute_Jv,
            layerid_to_mod,
            layer_collection,
        )

        self._v = v.to_dict()
        parameters = []
        output = dict()
        for layer_id, layer in layer_collection.layers.items():
            mod = layer_collection.get_layerid_module_map(self.model)[layer_id]
            mod_class = mod.__class__.__name__
            if mod_class in ["BatchNorm1d", "BatchNorm2d"]:
                raise NotImplementedError
            parameters.append(mod.weight)
            output[mod.weight] = torch.zeros_like(mod.weight)
            if layer.has_bias():
                parameters.append(mod.bias)
                output[mod.bias] = torch.zeros_like(mod.bias)

        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)

        self.i_output = 0
        self.start = 0
        for d in loader:
            self.xs = dict()
            inputs = d[0]
            grad_wrt = self._infer_differentiable_leafs(inputs, layerid_to_mod.values())
            bs = inputs.size(0)

            f_output = self.function(*d).view(bs, -1)
            n_output = f_output.size(-1)
            for i in range(n_output):
                # TODO reuse instead of reallocating memory
                self._Jv = torch.zeros((1, bs), device=device, dtype=dtype)

                self.compute_switch = True
                torch.autograd.grad(
                    f_output[:, i].sum(dim=0),
                    grad_wrt,
                    retain_graph=True,
                    only_inputs=True,
                )
                self.compute_switch = False
                pseudo_loss = torch.dot(self._Jv[0, :], f_output[:, i])
                grads = torch.autograd.grad(
                    pseudo_loss,
                    parameters,
                    retain_graph=i < n_output - 1,
                    only_inputs=True,
                )
                for i_p, p in enumerate(parameters):
                    output[p].add_(grads[i_p])

        output_dict = dict()
        for layer_id, layer in layer_collection.layers.items():
            mod = layer_collection.get_layerid_module_map(self.model)[layer_id]
            if layer.has_bias():
                output_dict[layer_id] = (
                    output[mod.weight] / n_examples,
                    output[mod.bias] / n_examples,
                )
            else:
                output_dict[layer_id] = (output[mod.weight] / n_examples,)

        # remove hooks
        del self.xs
        del self._Jv
        del self._v
        del self.compute_switch
        for h in self.handles:
            h.remove()

        return PVector(layer_collection=layer_collection, dict_repr=output_dict)

    def implicit_vTMv(self, v, examples, layer_collection):
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        # add hooks
        self.handles += self._add_hooks(
            self._hook_savex,
            self._hook_compute_Jv,
            layerid_to_mod,
            layer_collection,
        )

        self._v = v.to_dict()

        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)

        for layer_id, layer in layer_collection.layers.items():
            mod = layer_collection.get_layerid_module_map(self.model)[layer_id]
            mod_class = mod.__class__.__name__
            if mod_class in ["BatchNorm1d", "BatchNorm2d"]:
                raise NotImplementedError

        self.i_output = 0
        self.start = 0
        norm2 = 0
        self.compute_switch = True
        for d in loader:
            self.xs = dict()
            inputs = d[0]
            grad_wrt = self._infer_differentiable_leafs(inputs, layerid_to_mod.values())
            bs = inputs.size(0)

            f_output = self.function(*d).view(bs, -1).sum(dim=0)
            n_output = f_output.size(-1)
            for i in range(n_output):
                # TODO reuse instead of reallocating memory
                self._Jv = torch.zeros((1, bs), device=device, dtype=dtype)

                torch.autograd.grad(
                    f_output[i],
                    grad_wrt,
                    retain_graph=i < n_output - 1,
                    only_inputs=True,
                )
                norm2 += (self._Jv**2).sum()
        norm = norm2 / n_examples

        # remove hooks
        del self.xs
        del self._Jv
        del self._v
        del self.compute_switch
        for h in self.handles:
            h.remove()

        return norm

    def implicit_trace(self, examples, layer_collection):
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        # add hooks
        self.handles += self._add_hooks(
            self._hook_savex,
            self._hook_compute_trace,
            layerid_to_mod,
            layer_collection,
        )

        device = next(self.model.parameters()).device
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)

        self._trace = torch.tensor(0.0, device=device)
        for d in loader:
            self.xs = dict()
            inputs = d[0]
            grad_wrt = self._infer_differentiable_leafs(inputs, layerid_to_mod.values())
            bs = inputs.size(0)
            output = self.function(*d).view(bs, -1).sum(dim=0)
            n_output = output.size(-1)
            for i in range(n_output):
                torch.autograd.grad(
                    output[i],
                    grad_wrt,
                    retain_graph=i < n_output - 1,
                    only_inputs=True,
                )
        trace = self._trace / n_examples

        # remove hooks
        del self.xs
        del self._trace
        for h in self.handles:
            h.remove()

        return trace

    def implicit_Jv(self, v, examples, layer_collection):
        layerid_to_mod = layer_collection.get_layerid_module_map(self.model)
        # add hooks
        self.handles += self._add_hooks(
            self._hook_savex,
            self._hook_compute_Jv,
            layerid_to_mod,
            layer_collection,
        )

        self._v = v.to_dict()

        device = self._check_same_device(layerid_to_mod.values())
        dtype = self._check_same_dtype(layerid_to_mod.values())
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)

        def _f(n_output):
            self._Jv = torch.zeros((n_output, n_examples), device=device, dtype=dtype)

        self.delayed_for_n_ouput.append(_f)

        self.start = 0
        self.compute_switch = True
        for d in loader:
            self.xs = dict()
            inputs = d[0]
            grad_wrt = self._infer_differentiable_leafs(inputs, layerid_to_mod.values())
            bs = inputs.size(0)
            output = self.function(*d).view(bs, -1).sum(dim=0)
            n_output = output.size(-1)
            self._exec_delayed_n_output(n_output)
            for self.i_output in range(n_output):
                retain_graph = self.i_output < n_output - 1
                torch.autograd.grad(
                    output[self.i_output],
                    grad_wrt,
                    retain_graph=retain_graph,
                    only_inputs=True,
                )
            self.start += inputs.size(0)
        Jv = self._Jv

        # remove hooks
        del self.xs
        del self._Jv
        del self._v
        del self.start
        del self.i_output
        del self.compute_switch
        for h in self.handles:
            h.remove()

        return FVector(vector_repr=Jv)

    def _add_hooks(self, hook_x, hook_gy, layerid_to_mod, layer_collection):
        handles = []

        def _hook_x(mod, i, o, layer_id):
            hook_x(mod, i)
            o.register_hook(
                lambda g_o: hook_gy(
                    mod, g_o, layer_id=layer_id, layer_collection=layer_collection
                )
            )

        for l_id, mod in layerid_to_mod.items():
            handles.append(mod.register_forward_hook(partial(_hook_x, layer_id=l_id)))
        return handles

    def _hook_savex(self, mod, i):
        if mod in self.xs:
            raise NotImplementedError(
                """Passing through the same layer twice, this is not
                currently implemented with TorchHook backend"""
            )
        self.xs[mod] = i[0]

    def _hook_savex_io(self, mod, i):
        if self.outerloop_switch:
            self.x_outer[mod] = i[0]
        else:
            self.x_inner[mod] = i[0]

    def _hook_compute_flat_grad(self, mod, gy, layer_id, layer_collection):
        x = self.xs[mod]
        bs = x.size(0)
        layer = layer_collection[layer_id]
        start_p = layer_collection.p_pos[layer_id]
        FactoryMap[layer.__class__].flat_grad(
            self.grads[
                self.i_output,
                self.start : self.start + bs,
                start_p : start_p + layer.numel(),
            ],
            mod,
            layer,
            x,
            gy,
        )

    def _hook_compute_diag(self, mod, gy, layer_id, layer_collection):
        x = self.xs[mod]
        layer = layer_collection[layer_id]
        start_p = layer_collection.p_pos[layer_id]
        FactoryMap[layer.__class__].diag(
            self.diag_m[start_p : start_p + layer.numel()], mod, layer, x, gy
        )

    def _hook_compute_quasidiag(self, mod, gy, layer_id, layer_collection):
        x = self.xs[mod]
        layer = layer_collection[layer_id]
        diag, cross = self._blocks[layer_id]
        FactoryMap[layer.__class__].quasidiag(diag, cross, mod, layer, x, gy)

    def _hook_compute_layer_blocks(self, mod, gy, layer_id, layer_collection):
        x = self.xs[mod]
        layer = layer_collection[layer_id]
        block = self._blocks[layer_id]
        FactoryMap[layer.__class__].layer_block(block, mod, layer, x, gy)

    def _hook_compute_kfac_blocks(self, mod, gy, layer_id, layer_collection):
        mod_class = mod.__class__.__name__
        x = self.xs[mod]
        layer = layer_collection[layer_id]
        block = self._blocks[layer_id]
        if mod_class in ["Linear", "Conv2d", "Conv1d", "Embedding"]:
            FactoryMap[layer.__class__].kfac_gg(block[1], mod, layer, x, gy)
            if self.i_output == 0:
                # do this only once if n_output > 1
                FactoryMap[layer.__class__].kfac_xx(block[0], mod, layer, x, gy)
        else:
            raise NotImplementedError

    def _hook_compute_kfe_diag(self, mod, gy, layer_id, layer_collection):
        mod_class = mod.__class__.__name__
        layer = layer_collection[layer_id]
        x = self.xs[mod]
        evecs_a, evecs_g = self._kfe[layer_id]
        if mod_class in ["Linear", "Conv2d", "Conv1d", "Embedding"]:
            FactoryMap[layer.__class__].kfe_diag(
                self._diags[layer_id], mod, layer, x, gy, evecs_a, evecs_g
            )
        else:
            raise NotImplementedError

    def _hook_kxy(self, mod, gy, layer_id, layer_collection):
        if self.outerloop_switch:
            self.gy_outer[mod] = gy
        else:
            layer = layer_collection[layer_id]
            gy_inner = gy
            gy_outer = self.gy_outer[mod]
            x_outer = self.x_outer[mod]
            x_inner = self.x_inner[mod]
            bs_inner = x_inner.size(0)
            bs_outer = x_outer.size(0)
            FactoryMap[layer.__class__].kxy(
                self.G[
                    self.i_output_inner,
                    self.e_inner : self.e_inner + bs_inner,
                    self.i_output_outer,
                    self.e_outer : self.e_outer + bs_outer,
                ],
                mod,
                layer,
                x_inner,
                gy_inner,
                x_outer,
                gy_outer,
            )

    def _hook_compute_Jv(self, mod, gy, layer_id, layer_collection):
        if self.compute_switch:
            x = self.xs[mod]
            bs = x.size(0)
            layer = layer_collection.layers[layer_id]
            v_weight = self._v[layer_id][0]
            v_bias = None
            if layer.has_bias():
                v_bias = self._v[layer_id][1]
            FactoryMap[layer.__class__].Jv(
                self._Jv[self.i_output, self.start : self.start + bs],
                mod,
                layer,
                x,
                gy,
                v_weight,
                v_bias,
            )

    def _hook_compute_trace(self, mod, gy, layer_id, layer_collection):
        x = self.xs[mod]
        layer = layer_collection.layers[layer_id]
        FactoryMap[layer.__class__].trace(self._trace, mod, layer, x, gy)

    def _exec_delayed_n_output(self, n_output):

        while len(self.delayed_for_n_ouput) > 0:
            self.delayed_for_n_ouput.pop()(n_output)
