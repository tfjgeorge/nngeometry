import torch
import torch.nn.functional as F
from ..utils import (get_individual_modules, per_example_grad_conv,
                     get_n_parameters)
from ..vector import FVector


class M2Gradients:
    def __init__(self, model, dataloader, loss_function):
        self.model = model
        self.dataloader = dataloader
        self.handles = []
        self.xs = dict()
        # maps parameters to their position in flattened representation
        self.mods, self.p_pos = get_individual_modules(model)
        self.loss_function = loss_function

    def get_n_parameters(self):
        return get_n_parameters(self.model)

    def get_device(self):
        return next(self.model.parameters()).device

    def get_matrix(self):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex,
                                        self._hook_compute_flat_grad)

        device = next(self.model.parameters()).device
        n_examples = len(self.dataloader.sampler)
        n_parameters = sum([p.numel() for p in self.model.parameters()])
        self.grads = torch.zeros((n_examples, n_parameters), device=device)
        self.start = 0
        for (inputs, targets) in self.dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            loss = self.loss_function(inputs, targets).sum()
            torch.autograd.grad(loss, [inputs])
            self.start += inputs.size(0)
        grads = self.grads

        # remove hooks
        del self.grads
        self.xs = dict()
        for h in self.handles:
            h.remove()

        return grads

    def implicit_Jv(self, v):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex,
                                        self._hook_compute_vTg)

        i = 0
        self._v = dict()
        for p in self.model.parameters():
            self._v[p] = v[i:i+p.numel()].view(*p.size())
            i += p.numel()

        device = next(self.model.parameters()).device
        n_examples = len(self.dataloader.sampler)
        self.compute_switch = True
        self._vTg = torch.zeros(n_examples, device=device)
        self.start = 0
        for (inputs, targets) in self.dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            loss = self.loss_function(inputs, targets).sum()
            torch.autograd.grad(loss, [inputs])
            self.start += inputs.size(0)
        vTg = self._vTg

        # remove hooks
        self.xs = dict()
        del self._vTg
        del self._v
        del self.start
        del self.compute_switch
        for h in self.handles:
            h.remove()

        return FVector(self.model, vector_repr=vTg)

    def _add_hooks(self, hook_x, hook_gy):
        handles = []
        for m in self.mods:
            handles.append(m.register_forward_pre_hook(hook_x))
            handles.append(m.register_backward_hook(hook_gy))
        return handles

    def _hook_savex(self, mod, i):
        self.xs[mod] = i[0]

    def _hook_compute_flat_grad(self, mod, grad_input, grad_output):
        mod_class = mod.__class__.__name__
        gy = grad_output[0]
        x = self.xs[mod]
        bs = x.size(0)
        start_p = self.p_pos[mod]
        if mod_class == 'Linear':
            self.grads[self.start:self.start+bs,
                       start_p:start_p+mod.weight.numel()] \
                .add_(torch.bmm(gy.unsqueeze(2), x.unsqueeze(1)).view(bs, -1))
            if mod.bias is not None:
                start_p += mod.weight.numel()
                self.grads[self.start:self.start+bs,
                           start_p:start_p+mod.bias.numel()] \
                    .add_(gy)
        elif mod_class == 'Conv2d':
            indiv_gw = per_example_grad_conv(mod, x, gy)
            self.grads[self.start:self.start+bs,
                       start_p:start_p+mod.weight.numel()] \
                .add_(indiv_gw.view(bs, -1))
            if mod.bias is not None:
                start_p += mod.weight.numel()
                self.grads[self.start:self.start+bs,
                           start_p:start_p+mod.bias.numel()] \
                    .add_(gy.sum(dim=(2, 3)))
        elif mod_class == 'BatchNorm1d':
            # BN should be in eval mode
            assert not mod.training
            x_normalized = F.batch_norm(x, mod.running_mean,
                                        mod.running_var,
                                        None, None, False)
            self.grads[self.start:self.start+bs,
                       start_p:start_p+mod.weight.numel()] \
                .add_(gy * x_normalized)
            if mod.bias is not None:
                start_p += mod.weight.numel()
                self.grads[self.start:self.start+bs,
                           start_p:start_p+mod.bias.numel()] \
                    .add_(gy)
        else:
            raise NotImplementedError

    def _hook_compute_vTg(self, mod, grad_input, grad_output):
        if self.compute_switch:
            mod_class = mod.__class__.__name__
            gy = grad_output[0]
            x = self.xs[mod]
            bs = x.size(0)
            if mod_class == 'Linear':
                self._vTg[self.start:self.start+bs].add_(
                    (torch.mm(x, self._v[mod.weight].t()) * gy).sum(dim=1))
                if mod.bias is not None:
                    self._vTg[self.start:self.start+bs].add_(
                        torch.mv(gy, self._v[mod.bias]))
            elif mod_class == 'Conv2d':
                gy2 = F.conv2d(x, self._v[mod.weight], stride=mod.stride,
                               padding=mod.padding, dilation=mod.dilation)
                self._vTg[self.start:self.start+bs].add_(
                    (gy * gy2).view(bs, -1).sum(dim=1))
                if mod.bias is not None:
                    self._vTg[self.start:self.start+bs].add_(
                        torch.mv(gy.sum(dim=(2, 3)), self._v[mod.bias]))
            else:
                raise NotImplementedError
