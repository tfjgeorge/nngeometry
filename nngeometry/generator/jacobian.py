import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from ..utils import per_example_grad_conv
from ..object.vector import PVector, FVector
from ..layercollection import LayerCollection


class Jacobian:
    """
    Computes jacobians :math:`\mathbf{J}_{ijk}=\\frac{\partial f\left(x_{j}\\right)_{i}}{\delta\mathbf{w}_{k}}`,
    FIM matrices :math:`\mathbf{F}_{k,k'}=\\frac{1}{n}\sum_{i,j}\mathbf{J}_{ijk}\mathbf{J}_{ijk'}`
    and NTK matrices :math:`\mathbf{K}_{iji'j'}=\sum_{k}\mathbf{J}_{ijk}\mathbf{J}_{ijk'}`.
    
    This generator is written in pure PyTorch and exploits some tricks in order to make computations
    more efficient.

    :param layer_collection:
    :type layer_collection: :class:`.layercollection.LayerCollection`
    :param model:
    :type model: Pytorch `nn.Module`
    :param function: A function :math:`f\left(X,Y,Z\\right)` where :math:`X,Y,Z` are minibatchs
        returned by the dataloader (Note that in some cases :math:`Y,Z` are not required). If None,
        it defaults to `function = lambda *x: model(x[0])`
    :type function: python function 
    :param n_output: How many output is there for each example of your function. E.g. in 10 class
        classification this would probably be 10.
    :type n_output: integer

    """
    def __init__(self, model, function=None, n_output=1,
                 centering=False, layer_collection=None):
        self.model = model
        self.handles = []
        self.xs = dict()
        self.n_output = n_output
        self.centering = centering

        if function is None:
            function = lambda *x: model(x[0])
        self.function = function

        if layer_collection is None:
            self.layer_collection = LayerCollection.from_model(model)
        else:
            self.layer_collection = layer_collection
        # maps parameters to their position in flattened representation
        self.l_to_m, self.m_to_l = \
            self.layer_collection.get_layerid_module_maps(model)

    def get_device(self):
        return next(self.model.parameters()).device

    def get_covariance_matrix(self, examples):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex,
                                        self._hook_compute_flat_grad,
                                        self.l_to_m.values())

        device = next(self.model.parameters()).device
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        n_parameters = self.layer_collection.numel()
        bs = loader.batch_size
        G = torch.zeros((n_parameters, n_parameters), device=device)
        self.grads = torch.zeros((1, bs, n_parameters), device=device)
        if self.centering:
            grad_mean = torch.zeros((self.n_output, n_parameters),
                                    device=device)

        self.start = 0
        self.i_output = 0
        for d in loader:
            inputs = d[0]
            inputs.requires_grad = True
            bs = inputs.size(0)
            output = self.function(*d).view(bs, self.n_output) \
                .sum(dim=0)
            for i in range(self.n_output):
                self.grads.zero_()
                torch.autograd.grad(output[i], [inputs],
                                    retain_graph=i < self.n_output - 1,
                                    only_inputs=True)
                G += torch.mm(self.grads[0].t(), self.grads[0])
                if self.centering:
                    grad_mean[i].add_(self.grads[0].sum(dim=0))
        G /= n_examples
        if self.centering:
            grad_mean /= n_examples
            G -= torch.mm(grad_mean.t(), grad_mean)

        # remove hooks
        del self.grads
        self.xs = dict()
        for h in self.handles:
            h.remove()

        return G

    def get_covariance_diag(self, examples):
        if self.centering:
            raise NotImplementedError
        # add hooks
        self.handles += self._add_hooks(self._hook_savex,
                                        self._hook_compute_diag,
                                        self.l_to_m.values())

        device = next(self.model.parameters()).device
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        n_parameters = self.layer_collection.numel()
        self.diag_m = torch.zeros((n_parameters,), device=device)
        self.start = 0
        for d in loader:
            inputs = d[0]
            inputs.requires_grad = True
            bs = inputs.size(0)
            output = self.function(*d).view(bs, self.n_output) \
                .sum(dim=0)
            for i in range(self.n_output):
                torch.autograd.grad(output[i], [inputs],
                                    retain_graph=i < self.n_output - 1,
                                    only_inputs=True)
        diag_m = self.diag_m / n_examples

        # remove hooks
        del self.diag_m
        self.xs = dict()
        for h in self.handles:
            h.remove()

        return diag_m

    def get_covariance_quasidiag(self, examples):
        if self.centering:
            raise NotImplementedError
        # add hooks
        self.handles += self._add_hooks(self._hook_savex,
                                        self._hook_compute_quasidiag,
                                        self.l_to_m.values())

        device = next(self.model.parameters()).device
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        self._blocks = dict()
        for layer_id, layer in self.layer_collection.layers.items():
            s = layer.numel()
            if layer.bias is None:
                self._blocks[layer_id] = (torch.zeros((s, ), device=device),
                                          None)
            else:
                cross_s = layer.weight.size
                self._blocks[layer_id] = (torch.zeros((s, ), device=device),
                                          torch.zeros(cross_s, device=device))

        for d in loader:
            inputs = d[0]
            inputs.requires_grad = True
            bs = inputs.size(0)
            output = self.function(*d).view(bs, self.n_output) \
                .sum(dim=0)
            for i in range(self.n_output):
                torch.autograd.grad(output[i], [inputs],
                                    retain_graph=i < self.n_output - 1,
                                    only_inputs=True)
        for d, c in self._blocks.values():
            d.div_(n_examples)
            if c is not None:
                c.div_(n_examples)

        blocks = self._blocks

        # remove hooks
        del self._blocks
        self.xs = dict()
        for h in self.handles:
            h.remove()

        return blocks

    def get_covariance_layer_blocks(self, examples):
        if self.centering:
            raise NotImplementedError
        # add hooks
        self.handles += self._add_hooks(self._hook_savex,
                                        self._hook_compute_layer_blocks,
                                        self.l_to_m.values())

        device = next(self.model.parameters()).device
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        self._blocks = dict()
        for layer_id, layer in self.layer_collection.layers.items():
            s = layer.numel()
            self._blocks[layer_id] = torch.zeros((s, s), device=device)

        for d in loader:
            inputs = d[0]
            inputs.requires_grad = True
            bs = inputs.size(0)
            output = self.function(*d).view(bs, self.n_output) \
                .sum(dim=0)
            for i in range(self.n_output):
                torch.autograd.grad(output[i], [inputs],
                                    retain_graph=i < self.n_output - 1,
                                    only_inputs=True)
        blocks = {m: self._blocks[m] / n_examples for m in self._blocks.keys()}

        # remove hooks
        del self._blocks
        self.xs = dict()
        for h in self.handles:
            h.remove()

        return blocks

    def get_kfac_blocks(self, examples):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex,
                                        self._hook_compute_kfac_blocks,
                                        self.l_to_m.values())

        device = next(self.model.parameters()).device
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        self._blocks = dict()
        for layer_id, layer in self.layer_collection.layers.items():
            layer_class = layer.__class__.__name__
            if layer_class == 'LinearLayer':
                sG = layer.out_features
                sA = layer.in_features
            elif layer_class == 'Conv2dLayer':
                sG = layer.out_channels
                sA = layer.in_channels * layer.kernel_size[0] * \
                    layer.kernel_size[1]
            if layer.bias is not None:
                sA += 1
            self._blocks[layer_id] = (torch.zeros((sA, sA), device=device),
                                      torch.zeros((sG, sG), device=device))

        for d in loader:
            inputs = d[0]
            inputs.requires_grad = True
            bs = inputs.size(0)
            output = self.function(*d).view(bs, self.n_output) \
                .sum(dim=0)
            for self.i_output in range(self.n_output):
                retain_graph = self.i_output < self.n_output - 1
                torch.autograd.grad(output[self.i_output], [inputs],
                                    retain_graph=retain_graph,
                                    only_inputs=True)
        for layer_id in self.layer_collection.layers.keys():
            self._blocks[layer_id][0].div_(n_examples / self.n_output**.5)
            self._blocks[layer_id][1].div_(self.n_output**.5 * n_examples)
        blocks = self._blocks
        # blocks = {layer_id: (self._blocks[layer_id][0] / n_examples *
        #                      self.n_output**.5,
        #                      self._blocks[layer_id][1] / n_examples /
        #                      self.n_output**.5)
        #           for layer_id in self.layer_collection.layers.keys()}

        # remove hooks
        del self._blocks
        del self.i_output
        self.xs = dict()
        for h in self.handles:
            h.remove()

        return blocks

    def get_jacobian(self, examples):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex,
                                        self._hook_compute_flat_grad,
                                        self.l_to_m.values())

        device = next(self.model.parameters()).device
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        n_parameters = self.layer_collection.numel()
        self.grads = torch.zeros((self.n_output, n_examples, n_parameters),
                                 device=device)
        self.start = 0
        for d in loader:
            inputs = d[0]
            inputs.requires_grad = True
            bs = inputs.size(0)
            output = self.function(*d).view(bs, self.n_output) \
                .sum(dim=0)
            for self.i_output in range(self.n_output):
                retain_graph = self.i_output < self.n_output - 1
                torch.autograd.grad(output[self.i_output], [inputs],
                                    retain_graph=retain_graph,
                                    only_inputs=True)
            self.start += inputs.size(0)
        grads = self.grads
        if self.centering:
            grads -= grads.mean(dim=1, keepdim=True)

        # remove hooks
        del self.grads
        del self.start
        del self.i_output
        self.xs = dict()
        for h in self.handles:
            h.remove()

        return grads

    def get_gram_matrix(self, examples):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex_io, self._hook_kxy,
                                        self.l_to_m.values())

        device = next(self.model.parameters()).device
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        self.G = torch.zeros((self.n_output, n_examples,
                              self.n_output, n_examples), device=device)
        self.x_outer = dict()
        self.x_inner = dict()
        self.gy_outer = dict()
        self.e_outer = 0
        for i_outer, d in enumerate(loader):
            # used in hooks to switch between store/compute
            inputs_outer = d[0]
            inputs_outer.requires_grad = True
            bs_outer = inputs_outer.size(0)
            self.outerloop_switch = True
            output_outer = self.function(*d).view(bs_outer, self.n_output) \
                .sum(dim=0)
            for self.i_output_outer in range(self.n_output):
                self.outerloop_switch = True
                torch.autograd.grad(output_outer[self.i_output_outer],
                                    [inputs_outer], retain_graph=True,
                                    only_inputs=True)
                self.outerloop_switch = False

                self.e_inner = 0
                for i_inner, d in enumerate(loader):
                    if i_inner > i_outer:
                        break
                    inputs_inner = d[0]
                    inputs_inner.requires_grad = True
                    bs_inner = inputs_inner.size(0)
                    output_inner = self.function(*d).view(bs_inner,
                                                          self.n_output) \
                        .sum(dim=0)
                    for self.i_output_inner in range(self.n_output):
                        torch.autograd.grad(output_inner[self.i_output_inner],
                                            [inputs_inner], retain_graph=True,
                                            only_inputs=True)

                    # since self.G is a symmetric matrix we only need to
                    # compute the upper or lower triangle
                    # => copy block and exclude diagonal
                    if (i_inner < i_outer and
                            self.i_output_outer == self.n_output - 1):
                        self.G[:, self.e_outer:self.e_outer+bs_outer, :,
                               self.e_inner:self.e_inner+bs_inner] += \
                            self.G[:, self.e_inner:self.e_inner+bs_inner, :,
                                   self.e_outer:self.e_outer+bs_outer] \
                                .permute(2, 3, 0, 1)
                    self.e_inner += inputs_inner.size(0)

            self.e_outer += inputs_outer.size(0)
        G = self.G
        if self.centering:
            C = torch.eye(n_examples, device=G.device) - \
                torch.ones((n_examples, n_examples), device=G.device) / \
                n_examples
            sG = G.size()
            G = torch.mm(G.view(-1, n_examples), C)
            G = torch.mm(C, G.view(sG[0], sG[1], -1).permute(1, 0, 2)
                         .contiguous().view(n_examples, -1)) \
                .view(sG[1], sG[0], -1).permute(1, 0, 2).contiguous().view(*sG)

        # remove hooks
        del self.e_inner, self.e_outer
        del self.G
        del self.x_inner
        del self.x_outer
        del self.gy_outer
        for h in self.handles:
            h.remove()

        return G

    def get_kfe_diag(self, kfe, examples):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex,
                                        self._hook_compute_kfe_diag,
                                        self.l_to_m.values())

        device = next(self.model.parameters()).device
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        self._diags = dict()
        self._kfe = kfe
        for layer_id, layer in self.layer_collection.layers.items():
            layer_class = layer.__class__.__name__
            if layer_class == 'LinearLayer':
                sG = layer.out_features
                sA = layer.in_features
            elif layer_class == 'Conv2dLayer':
                sG = layer.out_channels
                sA = layer.in_channels * layer.kernel_size[0] * \
                    layer.kernel_size[1]
            if layer.bias is not None:
                sA += 1
            self._diags[layer_id] = torch.zeros((sG * sA), device=device)

        for d in loader:
            inputs = d[0]
            inputs.requires_grad = True
            bs = inputs.size(0)
            output = self.function(*d).view(bs, self.n_output) \
                .sum(dim=0)
            for self.i_output in range(self.n_output):
                retain_graph = self.i_output < self.n_output - 1
                torch.autograd.grad(output[self.i_output], [inputs],
                                    retain_graph=retain_graph,
                                    only_inputs=True)
        diags = {l_id: self._diags[l_id] / n_examples
                 for l_id in self.layer_collection.layers.keys()}

        # remove hooks
        del self._diags
        del self._kfe
        self.xs = dict()
        for h in self.handles:
            h.remove()

        return diags

    def implicit_mv(self, v, examples):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex,
                                        self._hook_compute_Jv,
                                        self.l_to_m.values())

        self._v = v.get_dict_representation()
        parameters = []
        output = dict()
        for layer_id, layer in self.layer_collection.layers.items():
            mod = self.l_to_m[layer_id]
            mod_class = mod.__class__.__name__
            if mod_class in ['BatchNorm1d', 'BatchNorm2d']:
                raise NotImplementedError
            parameters.append(mod.weight)
            output[mod.weight] = torch.zeros_like(mod.weight)
            if layer.bias is not None:
                parameters.append(mod.bias)
                output[mod.bias] = torch.zeros_like(mod.bias)

        device = next(self.model.parameters()).device
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)

        self.i_output = 0
        self.start = 0
        for d in loader:
            inputs = d[0]
            inputs.requires_grad = True
            bs = inputs.size(0)

            f_output = self.function(*d).view(bs, self.n_output)
            for i in range(self.n_output):
                # TODO reuse instead of reallocating memory
                self._Jv = torch.zeros((1, bs), device=device)

                self.compute_switch = True
                torch.autograd.grad(f_output[:, i].sum(dim=0), [inputs],
                                    retain_graph=True,
                                    only_inputs=True)
                self.compute_switch = False
                pseudo_loss = torch.dot(self._Jv[0, :], f_output[:, i])
                grads = torch.autograd.grad(pseudo_loss,
                                            parameters,
                                            retain_graph=i < self.n_output - 1,
                                            only_inputs=True)
                for i_p, p in enumerate(parameters):
                    output[p].add_(grads[i_p])

        output_dict = dict()
        for layer_id, layer in self.layer_collection.layers.items():
            mod = self.l_to_m[layer_id]
            if layer.bias is None:
                output_dict[layer_id] = (output[mod.weight] / n_examples,)
            else:
                output_dict[layer_id] = (output[mod.weight] / n_examples,
                                         output[mod.bias] / n_examples)

        # remove hooks
        self.xs = dict()
        del self._Jv
        del self._v
        del self.compute_switch
        for h in self.handles:
            h.remove()

        return PVector(layer_collection=self.layer_collection,
                       dict_repr=output_dict)

    def implicit_vTMv(self, v, examples):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex,
                                        self._hook_compute_Jv,
                                        self.l_to_m.values())

        self._v = v.get_dict_representation()

        device = next(self.model.parameters()).device
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)

        for layer_id, layer in self.layer_collection.layers.items():
            mod = self.l_to_m[layer_id]
            mod_class = mod.__class__.__name__
            if mod_class in ['BatchNorm1d', 'BatchNorm2d']:
                raise NotImplementedError

        self.i_output = 0
        self.start = 0
        norm2 = 0
        self.compute_switch = True
        for d in loader:
            inputs = d[0]
            inputs.requires_grad = True
            bs = inputs.size(0)

            f_output = self.function(*d).view(bs, self.n_output).sum(dim=0)
            for i in range(self.n_output):
                # TODO reuse instead of reallocating memory
                self._Jv = torch.zeros((1, bs), device=device)

                torch.autograd.grad(f_output[i], [inputs],
                                    retain_graph=i < self.n_output - 1,
                                    only_inputs=True)
                norm2 += (self._Jv**2).sum()
        norm = norm2 / n_examples

        # remove hooks
        self.xs = dict()
        del self._Jv
        del self._v
        del self.compute_switch
        for h in self.handles:
            h.remove()

        return norm

    def implicit_trace(self, examples):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex,
                                        self._hook_compute_trace,
                                        self.l_to_m.values())

        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)

        self._trace = 0
        for d in loader:
            inputs = d[0]
            inputs.requires_grad = True
            bs = inputs.size(0)
            output = self.function(*d).view(bs, self.n_output) \
                .sum(dim=0)
            for i in range(self.n_output):
                torch.autograd.grad(output[i], [inputs],
                                    retain_graph=i < self.n_output - 1,
                                    only_inputs=True)
        trace = self._trace / n_examples

        # remove hooks
        self.xs = dict()
        del self._trace
        for h in self.handles:
            h.remove()

        return trace

    def implicit_Jv(self, v, examples):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex,
                                        self._hook_compute_Jv,
                                        self.l_to_m.values())

        self._v = v.get_dict_representation()

        device = next(self.model.parameters()).device
        loader = self._get_dataloader(examples)
        n_examples = len(loader.sampler)
        self._Jv = torch.zeros((self.n_output, n_examples), device=device)
        self.start = 0
        self.compute_switch = True
        for d in loader:
            inputs = d[0]
            inputs.requires_grad = True
            bs = inputs.size(0)
            output = self.function(*d).view(bs, self.n_output) \
                .sum(dim=0)
            for self.i_output in range(self.n_output):
                retain_graph = self.i_output < self.n_output - 1
                torch.autograd.grad(output[self.i_output], [inputs],
                                    retain_graph=retain_graph,
                                    only_inputs=True)
            self.start += inputs.size(0)
        Jv = self._Jv

        # remove hooks
        self.xs = dict()
        del self._Jv
        del self._v
        del self.start
        del self.i_output
        del self.compute_switch
        for h in self.handles:
            h.remove()

        return FVector(vector_repr=Jv)

    def _check_bn_training(self, mod):
        # check that BN layers are in eval mode
        if mod.training:
            raise NotImplementedError('I don\'t know what to do with BN ' +
                                      'layers in training mode')

    def _add_hooks(self, hook_x, hook_gy, mods):
        handles = []
        for m in mods:
            handles.append(m.register_forward_pre_hook(hook_x))
            handles.append(m.register_backward_hook(hook_gy))
        return handles

    def _hook_savex(self, mod, i):
        self.xs[mod] = i[0]

    def _hook_savex_io(self, mod, i):
        if self.outerloop_switch:
            self.x_outer[mod] = i[0]
        else:
            self.x_inner[mod] = i[0]

    def _hook_compute_flat_grad(self, mod, grad_input, grad_output):
        mod_class = mod.__class__.__name__
        gy = grad_output[0]
        x = self.xs[mod]
        bs = x.size(0)
        layer_id = self.m_to_l[mod]
        start_p = self.layer_collection.p_pos[layer_id]
        if mod_class == 'Linear':
            self.grads[self.i_output, self.start:self.start+bs,
                       start_p:start_p+mod.weight.numel()] \
                .add_(torch.bmm(gy.unsqueeze(2), x.unsqueeze(1)).view(bs, -1))
            if self.layer_collection[layer_id].bias is not None:
                start_p += mod.weight.numel()
                self.grads[self.i_output, self.start:self.start+bs,
                           start_p:start_p+mod.bias.numel()] \
                    .add_(gy)
        elif mod_class == 'Conv2d':
            indiv_gw = per_example_grad_conv(mod, x, gy)
            self.grads[self.i_output, self.start:self.start+bs,
                       start_p:start_p+mod.weight.numel()] \
                .add_(indiv_gw.view(bs, -1))
            if self.layer_collection[layer_id].bias is not None:
                start_p += mod.weight.numel()
                self.grads[self.i_output, self.start:self.start+bs,
                           start_p:start_p+mod.bias.numel()] \
                    .add_(gy.sum(dim=(2, 3)))
        elif mod_class == 'BatchNorm1d':
            self._check_bn_training(mod)
            x_normalized = F.batch_norm(x, mod.running_mean,
                                        mod.running_var,
                                        None, None, mod.training,
                                        momentum=0.)
            self.grads[self.i_output, self.start:self.start+bs,
                       start_p:start_p+mod.weight.numel()] \
                .add_(gy * x_normalized)
            start_p += mod.weight.numel()
            self.grads[self.i_output, self.start:self.start+bs,
                       start_p:start_p+mod.bias.numel()] \
                .add_(gy)
        elif mod_class == 'BatchNorm2d':
            self._check_bn_training(mod)
            x_normalized = F.batch_norm(x, mod.running_mean,
                                        mod.running_var,
                                        None, None, mod.training,
                                        momentum=0.)
            self.grads[self.i_output, self.start:self.start+bs,
                       start_p:start_p+mod.weight.numel()] \
                .add_((gy * x_normalized).sum(dim=(2, 3)))
            start_p += mod.weight.numel()
            self.grads[self.i_output, self.start:self.start+bs,
                       start_p:start_p+mod.bias.numel()] \
                .add_(gy.sum(dim=(2, 3)))
        elif mod_class == 'GroupNorm':
            x_normalized = F.group_norm(x, mod.num_groups,
                                        eps=mod.eps)
            self.grads[self.i_output, self.start:self.start+bs,
                       start_p:start_p+mod.weight.numel()] \
                .add_((gy * x_normalized).sum(dim=(2, 3)))
            start_p += mod.weight.numel()
            self.grads[self.i_output, self.start:self.start+bs,
                       start_p:start_p+mod.bias.numel()] \
                .add_(gy.sum(dim=(2, 3)))
        else:
            raise NotImplementedError

    def _hook_compute_diag(self, mod, grad_input, grad_output):
        mod_class = mod.__class__.__name__
        gy = grad_output[0]
        x = self.xs[mod]
        layer_id = self.m_to_l[mod]
        start_p = self.layer_collection.p_pos[layer_id]
        if mod_class == 'Linear':
            self.diag_m[start_p:start_p+mod.weight.numel()] \
                .add_(torch.mm(gy.t()**2, x**2).view(-1))
            if self.layer_collection[layer_id].bias is not None:
                start_p += mod.weight.numel()
                self.diag_m[start_p: start_p+mod.bias.numel()] \
                    .add_((gy**2).sum(dim=0))
        elif mod_class == 'Conv2d':
            indiv_gw = per_example_grad_conv(mod, x, gy)
            self.diag_m[start_p:start_p+mod.weight.numel()] \
                .add_((indiv_gw**2).sum(dim=0).view(-1))
            if self.layer_collection[layer_id].bias is not None:
                start_p += mod.weight.numel()
                self.diag_m[start_p:start_p+mod.bias.numel()] \
                    .add_((gy.sum(dim=(2, 3))**2).sum(dim=0))
        elif mod_class == 'BatchNorm1d':
            self._check_bn_training(mod)
            x_normalized = F.batch_norm(x, mod.running_mean,
                                        mod.running_var,
                                        None, None, mod.training)
            self.diag_m[start_p:start_p+mod.weight.numel()] \
                .add_((gy**2 * x_normalized**2).sum(dim=0).view(-1))
            start_p += mod.weight.numel()
            self.diag_m[start_p: start_p+mod.bias.numel()] \
                .add_((gy**2).sum(dim=0))
        elif mod_class == 'BatchNorm2d':
            self._check_bn_training(mod)
            x_normalized = F.batch_norm(x, mod.running_mean,
                                        mod.running_var,
                                        None, None, mod.training)
            self.diag_m[start_p:start_p+mod.weight.numel()] \
                .add_(((gy * x_normalized).sum(dim=(2, 3))**2).sum(dim=0)
                      .view(-1))
            start_p += mod.weight.numel()
            self.diag_m[start_p: start_p+mod.bias.numel()] \
                .add_((gy.sum(dim=(2, 3))**2).sum(dim=0))
        elif mod_class == 'GroupNorm':
            x_normalized = F.group_norm(x, mod.num_groups,
                                        None, None, eps=mod.eps)
            self.diag_m[start_p:start_p+mod.weight.numel()] \
                .add_(((gy * x_normalized).sum(dim=(2, 3))**2).sum(dim=0)
                      .view(-1))
            start_p += mod.weight.numel()
            self.diag_m[start_p: start_p+mod.bias.numel()] \
                .add_((gy.sum(dim=(2, 3))**2).sum(dim=0))
        else:
            raise NotImplementedError

    def _hook_compute_quasidiag(self, mod, grad_input, grad_output):
        mod_class = mod.__class__.__name__
        gy = grad_output[0]
        x = self.xs[mod]
        bs = x.size(0)
        layer_id = self.m_to_l[mod]
        diag, cross = self._blocks[layer_id]

        sw = self.layer_collection[layer_id].weight.numel()
        if mod_class == 'Linear':
            diag[:sw].add_(torch.mm(gy.t()**2, x**2).view(-1))
            if self.layer_collection[layer_id].bias is not None:
                diag[sw:].add_((gy**2).sum(dim=0))
                cross.add_(torch.mm(gy.t()**2, x))
        elif mod_class == 'Conv2d':
            indiv_gw = per_example_grad_conv(mod, x, gy)
            diag[:sw].add_((indiv_gw**2).sum(dim=0).view(-1))
            if self.layer_collection[layer_id].bias is not None:
                gb_per_example = gy.sum(dim=(2, 3))
                diag[sw:].add_((gb_per_example**2).sum(dim=0))
                y = (gy * gb_per_example.unsqueeze(2).unsqueeze(3))
                cross_this = F.conv2d(x.transpose(0, 1),
                                    y.transpose(0, 1),
                                    stride=mod.dilation,
                                    padding=mod.padding,
                                    dilation=mod.stride).transpose(0, 1)
                cross_this = cross_this[:, :, :mod.kernel_size[0], :mod.kernel_size[1]]
                cross.add_(cross_this)
        else:
            raise NotImplementedError

    def _hook_compute_layer_blocks(self, mod, grad_input, grad_output):
        mod_class = mod.__class__.__name__
        gy = grad_output[0]
        x = self.xs[mod]
        bs = x.size(0)
        layer_id = self.m_to_l[mod]
        block = self._blocks[layer_id]
        if mod_class == 'Linear':
            gw = torch.bmm(gy.unsqueeze(2), x.unsqueeze(1)).view(bs, -1)
            if self.layer_collection[layer_id].bias is not None:
                gw = torch.cat([gw.view(bs, -1), gy.view(bs, -1)], dim=1)
            block.add_(torch.mm(gw.t(), gw))
        elif mod_class == 'Conv2d':
            gw = per_example_grad_conv(mod, x, gy).view(bs, -1)
            if self.layer_collection[layer_id].bias is not None:
                gw = torch.cat([gw, gy.sum(dim=(2, 3)).view(bs, -1)], dim=1)
            block.add_(torch.mm(gw.t(), gw))
        elif mod_class == 'BatchNorm1d':
            self._check_bn_training(mod)
            x_normalized = F.batch_norm(x, mod.running_mean,
                                        mod.running_var,
                                        None, None, mod.training)
            gw = gy * x_normalized
            gw = torch.cat([gw, gy], dim=1)
            block.add_(torch.mm(gw.t(), gw))
        elif mod_class == 'BatchNorm2d':
            self._check_bn_training(mod)
            x_normalized = F.batch_norm(x, mod.running_mean,
                                        mod.running_var,
                                        None, None, mod.training)
            gw = (gy * x_normalized).sum(dim=(2, 3))
            gw = torch.cat([gw, gy.sum(dim=(2, 3))], dim=1)
            block.add_(torch.mm(gw.t(), gw))
        elif mod_class == 'GroupNorm':
            x_normalized = F.group_norm(x, mod.num_groups,
                                        None, None, mod.eps)
            gw = (gy * x_normalized).sum(dim=(2, 3))
            gw = torch.cat([gw, gy.sum(dim=(2, 3))], dim=1)
            block.add_(torch.mm(gw.t(), gw))
        else:
            raise NotImplementedError

    def _hook_compute_kfac_blocks(self, mod, grad_input, grad_output):
        mod_class = mod.__class__.__name__
        gy = grad_output[0]
        x = self.xs[mod]
        layer_id = self.m_to_l[mod]
        block = self._blocks[layer_id]
        if mod_class == 'Linear':
            block[1].add_(torch.mm(gy.t(), gy))
            if self.layer_collection[layer_id].bias is not None:
                x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
            if self.i_output == 0:
                # do this only once if n_output > 1
                block[0].add_(torch.mm(x.t(), x))
        elif mod_class == 'Conv2d':
            ks = (mod.weight.size(2), mod.weight.size(3))
            if self.i_output == 0:
                # do this only once if n_output > 1
                # A_tilda in KFC
                A_tilda = F.unfold(x, kernel_size=ks, stride=mod.stride,
                                   padding=mod.padding, dilation=mod.dilation)
                # A_tilda is bs * #locations x #parameters
                A_tilda = A_tilda.permute(0, 2, 1).contiguous() \
                    .view(-1, A_tilda.size(1))
                if self.layer_collection[layer_id].bias is not None:
                    A_tilda = torch.cat([A_tilda,
                                         torch.ones_like(A_tilda[:, :1])],
                                        dim=1)
                # Omega_hat in KFC
                block[0].add_(torch.mm(A_tilda.t(), A_tilda))
            spatial_locations = gy.size(2) * gy.size(3)
            os = gy.size(1)
            # DS_tilda in KFC
            DS_tilda = gy.permute(0, 2, 3, 1).contiguous().view(-1, os)
            block[1].add_(torch.mm(DS_tilda.t(), DS_tilda) / spatial_locations)
        else:
            raise NotImplementedError

    def _hook_kxy(self, mod, grad_input, grad_output):
        if self.outerloop_switch:
            self.gy_outer[mod] = grad_output[0]
        else:
            mod_class = mod.__class__.__name__
            layer_id = self.m_to_l[mod]
            gy_inner = grad_output[0]
            gy_outer = self.gy_outer[mod]
            x_outer = self.x_outer[mod]
            x_inner = self.x_inner[mod]
            bs_inner = x_inner.size(0)
            bs_outer = x_outer.size(0)
            if mod_class == 'Linear':
                self.G[self.i_output_inner,
                       self.e_inner:self.e_inner+bs_inner,
                       self.i_output_outer,
                       self.e_outer:self.e_outer+bs_outer] += \
                    torch.mm(x_inner, x_outer.t()) * \
                    torch.mm(gy_inner, gy_outer.t())
                if self.layer_collection[layer_id].bias is not None:
                    self.G[self.i_output_inner,
                           self.e_inner:self.e_inner+bs_inner,
                           self.i_output_outer,
                           self.e_outer:self.e_outer+bs_outer] += \
                        torch.mm(gy_inner, gy_outer.t())
            elif mod_class == 'Conv2d':
                indiv_gw_inner = per_example_grad_conv(mod, x_inner, gy_inner)
                indiv_gw_outer = per_example_grad_conv(mod, x_outer, gy_outer)
                self.G[self.i_output_inner,
                       self.e_inner:self.e_inner+bs_inner,
                       self.i_output_outer,
                       self.e_outer:self.e_outer+bs_outer] += \
                    torch.mm(indiv_gw_inner.view(bs_inner, -1),
                             indiv_gw_outer.view(bs_outer, -1).t())
                if self.layer_collection[layer_id].bias is not None:
                    self.G[self.i_output_inner,
                           self.e_inner:self.e_inner+bs_inner,
                           self.i_output_outer,
                           self.e_outer:self.e_outer+bs_outer] += \
                        torch.mm(gy_inner.sum(dim=(2, 3)),
                                 gy_outer.sum(dim=(2, 3)).t())
            elif mod_class == 'BatchNorm1d':
                self._check_bn_training(mod)
                x_norm_inner = F.batch_norm(x_inner, mod.running_mean,
                                            mod.running_var,
                                            None, None, mod.training,
                                            momentum=0.)
                x_norm_outer = F.batch_norm(x_outer, mod.running_mean,
                                            mod.running_var,
                                            None, None, mod.training,
                                            momentum=0.)
                indiv_gw_inner = x_norm_inner * gy_inner
                indiv_gw_outer = x_norm_outer * gy_outer
                self.G[self.i_output_inner,
                       self.e_inner:self.e_inner+bs_inner,
                       self.i_output_outer,
                       self.e_outer:self.e_outer+bs_outer] += \
                    torch.mm(indiv_gw_inner, indiv_gw_outer.t())
                self.G[self.i_output_inner,
                       self.e_inner:self.e_inner+bs_inner,
                       self.i_output_outer,
                       self.e_outer:self.e_outer+bs_outer] += \
                    torch.mm(gy_inner, gy_outer.t())
            elif mod_class == 'BatchNorm2d':
                self._check_bn_training(mod)
                x_norm_inner = F.batch_norm(x_inner, mod.running_mean,
                                            mod.running_var,
                                            None, None, mod.training,
                                            momentum=0.)
                x_norm_outer = F.batch_norm(x_outer, mod.running_mean,
                                            mod.running_var,
                                            None, None, mod.training,
                                            momentum=0.)
                indiv_gw_inner = (x_norm_inner * gy_inner).sum(dim=(2, 3))
                indiv_gw_outer = (x_norm_outer * gy_outer).sum(dim=(2, 3))
                self.G[self.i_output_inner,
                       self.e_inner:self.e_inner+bs_inner,
                       self.i_output_outer,
                       self.e_outer:self.e_outer+bs_outer] += \
                    torch.mm(indiv_gw_inner, indiv_gw_outer.t())
                self.G[self.i_output_inner,
                       self.e_inner:self.e_inner+bs_inner,
                       self.i_output_outer,
                       self.e_outer:self.e_outer+bs_outer] += \
                    torch.mm(gy_inner.sum(dim=(2, 3)),
                             gy_outer.sum(dim=(2, 3)).t())
            elif mod_class == 'GroupNorm':
                x_norm_inner = F.group_norm(x_inner, mod.num_groups,
                                            None, None,
                                            eps=mod.eps)
                x_norm_outer = F.group_norm(x_outer, mod.num_groups,
                                            None, None,
                                            eps=mod.eps)
                indiv_gw_inner = (x_norm_inner * gy_inner).sum(dim=(2, 3))
                indiv_gw_outer = (x_norm_outer * gy_outer).sum(dim=(2, 3))
                self.G[self.i_output_inner,
                       self.e_inner:self.e_inner+bs_inner,
                       self.i_output_outer,
                       self.e_outer:self.e_outer+bs_outer] += \
                    torch.mm(indiv_gw_inner, indiv_gw_outer.t())
                self.G[self.i_output_inner,
                       self.e_inner:self.e_inner+bs_inner,
                       self.i_output_outer,
                       self.e_outer:self.e_outer+bs_outer] += \
                    torch.mm(gy_inner.sum(dim=(2, 3)),
                             gy_outer.sum(dim=(2, 3)).t())
            else:
                raise NotImplementedError

    def _hook_compute_kfe_diag(self, mod, grad_input, grad_output):
        mod_class = mod.__class__.__name__
        gy = grad_output[0]
        layer_id = self.m_to_l[mod]
        x = self.xs[mod]
        evecs_a, evecs_g = self._kfe[layer_id]
        if mod_class == 'Linear':
            if mod.bias is not None:
                x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
            gy_kfe = torch.mm(gy, evecs_g)
            x_kfe = torch.mm(x, evecs_a)
            self._diags[layer_id].add_(torch.mm(gy_kfe.t()**2, x_kfe**2)
                                       .view(-1))
        elif mod_class == 'Conv2d':
            ks = (mod.weight.size(2), mod.weight.size(3))
            gy_s = gy.size()
            bs = gy_s[0]
            # project x to kfe
            x_unfold = F.unfold(x, kernel_size=ks, stride=mod.stride,
                                padding=mod.padding, dilation=mod.dilation)
            x_unfold_s = x_unfold.size()
            x_unfold = x_unfold.view(bs, x_unfold_s[1], -1).permute(0, 2, 1)\
                .contiguous().view(-1, x_unfold_s[1])
            if mod.bias is not None:
                x_unfold = torch.cat([x_unfold,
                                      torch.ones_like(x_unfold[:, :1])], dim=1)
            x_kfe = torch.mm(x_unfold, evecs_a)

            # project gy to kfe
            gy = gy.view(bs, gy_s[1], -1).permute(0, 2, 1).contiguous()
            gy_kfe = torch.mm(gy.view(-1, gy_s[1]), evecs_g)
            gy_kfe = gy_kfe.view(bs, -1, gy_s[1]).permute(0, 2, 1).contiguous()

            indiv_gw = torch.bmm(gy_kfe.view(bs, gy_s[1], -1),
                                 x_kfe.view(bs, -1, x_kfe.size(1)))
            self._diags[layer_id].add_((indiv_gw**2).sum(dim=0).view(-1))
        else:
            raise NotImplementedError

    def _hook_compute_Jv(self, mod, grad_input, grad_output):
        if self.compute_switch:
            mod_class = mod.__class__.__name__
            gy = grad_output[0]
            x = self.xs[mod]
            bs = x.size(0)
            layer_id = self.m_to_l[mod]
            layer = self.layer_collection.layers[layer_id]
            v_weight = self._v[layer_id][0]
            if layer.bias is not None:
                v_bias = self._v[layer_id][1]

            if mod_class == 'Linear':
                self._Jv[self.i_output, self.start:self.start+bs].add_(
                    (torch.mm(x, v_weight.t()) * gy).sum(dim=1))
                if self.layer_collection[layer_id].bias is not None:
                    self._Jv[self.i_output, self.start:self.start+bs].add_(
                        torch.mv(gy.contiguous(), v_bias))
            elif mod_class == 'Conv2d':
                gy2 = F.conv2d(x, v_weight, stride=mod.stride,
                               padding=mod.padding, dilation=mod.dilation)
                self._Jv[self.i_output, self.start:self.start+bs].add_(
                    (gy * gy2).view(bs, -1).sum(dim=1))
                if self.layer_collection[layer_id].bias is not None:
                    self._Jv[self.i_output, self.start:self.start+bs].add_(
                        torch.mv(gy.sum(dim=(2, 3)), v_bias))
            elif mod_class == 'BatchNorm1d':
                self._check_bn_training(mod)
                x_normalized = F.batch_norm(x, mod.running_mean,
                                            mod.running_var,
                                            None, None, mod.training,
                                            momentum=0.)
                self._Jv[self.i_output, self.start:self.start+bs].add_(
                    torch.mv(gy * x_normalized, v_weight))
                self._Jv[self.i_output, self.start:self.start+bs].add_(
                    torch.mv(gy.contiguous(), v_bias))
            elif mod_class == 'BatchNorm2d':
                self._check_bn_training(mod)
                x_normalized = F.batch_norm(x, mod.running_mean,
                                            mod.running_var,
                                            None, None, mod.training,
                                            momentum=0.)
                self._Jv[self.i_output, self.start:self.start+bs].add_(
                    torch.mv((gy * x_normalized).sum(dim=(2, 3)),
                             v_weight))
                self._Jv[self.i_output, self.start:self.start+bs].add_(
                    torch.mv(gy.sum(dim=(2, 3)), v_bias))
            elif mod_class == 'GroupNorm':
                x_normalized = F.group_norm(x, mod.num_groups,
                                            None, None, mod.eps)
                self._Jv[self.i_output, self.start:self.start+bs].add_(
                    torch.mv((gy * x_normalized).sum(dim=(2, 3)),
                             v_weight))
                self._Jv[self.i_output, self.start:self.start+bs].add_(
                    torch.mv(gy.sum(dim=(2, 3)), v_bias))
            else:
                raise NotImplementedError

    def _hook_compute_trace(self, mod, grad_input, grad_output):
        mod_class = mod.__class__.__name__
        gy = grad_output[0]
        x = self.xs[mod]
        if mod_class == 'Linear':
            self._trace += torch.mm(gy.t()**2, x**2).sum()
            if mod.bias is not None:
                self._trace += (gy**2).sum()
        elif mod_class == 'Conv2d':
            indiv_gw = per_example_grad_conv(mod, x, gy)
            self._trace += (indiv_gw**2).sum()
            if mod.bias is not None:
                self._trace += (gy.sum(dim=(2, 3))**2).sum()
        elif mod_class == 'BatchNorm1d':
            self._check_bn_training(mod)
            x_normalized = F.batch_norm(x, mod.running_mean,
                                        mod.running_var,
                                        None, None, mod.training)
            self._trace += (gy**2 * x_normalized**2).sum()
            self._trace += (gy**2).sum()
        elif mod_class == 'BatchNorm2d':
            self._check_bn_training(mod)
            x_normalized = F.batch_norm(x, mod.running_mean,
                                        mod.running_var,
                                        None, None, mod.training)
            self._trace += ((gy * x_normalized).sum(dim=(2, 3))**2).sum()
            self._trace += (gy.sum(dim=(2, 3))**2).sum()
        elif mod_class == 'GroupNorm':
            x_normalized = F.group_norm(x, mod.num_groups,
                                        None, None, mod.eps)
            self._trace += ((gy * x_normalized).sum(dim=(2, 3))**2).sum()
            self._trace += (gy.sum(dim=(2, 3))**2).sum()
        else:
            raise NotImplementedError

    def _get_dataloader(self, examples):
        if isinstance(examples, DataLoader):
            return examples
        else:
            return DataLoader(TensorDataset(*examples),
                              batch_size=len(examples[0]))