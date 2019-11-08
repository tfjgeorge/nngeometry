import torch
import torch.nn.functional as F
from ..utils import get_individual_modules, per_example_grad_conv

class L2Loss:
    def __init__(self, model, dataloader, loss_closure):
        self.model = model
        self.dataloader = dataloader
        self.handles = []
        self.xs = dict()
        # self.p_pos maps parameters to their position in flattened representation
        self.mods, self.p_pos = get_individual_modules(model)
        self.loss_closure = loss_closure

    def get_n_parameters(self):
        return sum([p.numel() for p in self.model.parameters()])

    def get_device(self):
        return next(self.model.parameters()).device

    def get_matrix(self):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex, self._hook_compute_flat_grad)

        device = next(self.model.parameters()).device
        n_examples = len(self.dataloader.sampler)
        n_parameters = sum([p.numel() for p in self.model.parameters()])
        bs = self.dataloader.batch_size
        G = torch.zeros((n_parameters, n_parameters), device=device)
        self.grads = torch.zeros((bs, n_parameters), device=device)
        self.start = 0
        for (inputs, targets) in self.dataloader:
            self.grads.zero_()
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            loss = self.loss_closure(inputs, targets)
            torch.autograd.grad(loss, [inputs])
            G += torch.mm(self.grads.t(), self.grads)
        G /= n_examples

        # remove hooks
        del self.grads
        self.xs = dict()
        for h in self.handles:
            h.remove()

        return G

    def get_layer_blocks(self):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex, self._hook_compute_layer_blocks)

        device = next(self.model.parameters()).device
        n_examples = len(self.dataloader.sampler)
        self._blocks = dict()
        for m in self.mods:
            s = m.weight.numel()
            if m.bias is not None:
                s += m.bias.numel()
            self._blocks[m] = torch.zeros((s, s), device=device)

        for (inputs, targets) in self.dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            loss = self.loss_closure(inputs, targets)
            torch.autograd.grad(loss, [inputs])
        blocks = {m: self._blocks[m] / n_examples for m in self.mods}

        # remove hooks
        del self._blocks
        self.xs = dict()
        for h in self.handles:
            h.remove()

        return blocks

    def get_kfac_blocks(self):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex, self._hook_compute_kfac_blocks)

        device = next(self.model.parameters()).device
        n_examples = len(self.dataloader.sampler)
        self._blocks = dict()
        for m in self.mods:
            sG = m.weight.size(0)
            sA = m.weight.size(1)
            if m.bias is not None:
                sA += 1
            self._blocks[m] = (torch.zeros((sA, sA), device=device),
                               torch.zeros((sG, sG), device=device))

        for (inputs, targets) in self.dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            loss = self.loss_closure(inputs, targets)
            torch.autograd.grad(loss, [inputs])
        blocks = {m: (self._blocks[m][0] / n_examples, self._blocks[m][1] / n_examples)
                  for m in self.mods}

        # remove hooks
        del self._blocks
        self.xs = dict()
        for h in self.handles:
            h.remove()

        return blocks

    def get_diag(self):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex, self._hook_compute_diag)

        device = next(self.model.parameters()).device
        n_examples = len(self.dataloader.sampler)
        n_parameters = sum([p.numel() for p in self.model.parameters()])
        bs = self.dataloader.batch_size
        G = torch.zeros((n_parameters, n_parameters), device=device)
        self.diag_m = torch.zeros((n_parameters,), device=device)
        self.start = 0
        for (inputs, targets) in self.dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            loss = self.loss_closure(inputs, targets)
            torch.autograd.grad(loss, [inputs])
        diag_m = self.diag_m / n_examples

        # remove hooks
        del self.diag_m
        self.xs = dict()
        for h in self.handles:
            h.remove()

        return diag_m

    def get_lowrank_matrix(self):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex, self._hook_compute_flat_grad)

        device = next(self.model.parameters()).device
        n_examples = len(self.dataloader.sampler)
        n_parameters = sum([p.numel() for p in self.model.parameters()])
        bs = self.dataloader.batch_size
        G = torch.zeros((n_parameters, n_parameters), device=device)
        self.grads = torch.zeros((n_examples, n_parameters), device=device)
        self.start = 0
        for (inputs, targets) in self.dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            loss = self.loss_closure(inputs, targets)
            torch.autograd.grad(loss, [inputs])
            self.start += inputs.size(0)
        half_mat = self.grads / n_examples**.5

        # remove hooks
        del self.grads
        self.xs = dict()
        for h in self.handles:
            h.remove()

        return half_mat

    def implicit_mv(self, v):
        raise NotImplementedError

    def implicit_m_norm(self, v):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex, self._hook_compute_m_norm)

        i = 0
        self._v = dict()
        for p in self.model.parameters():
            self._v[p] = v[i:i+p.numel()].view(*p.size())
            i += p.numel()

        device = next(self.model.parameters()).device
        n_examples = len(self.dataloader.sampler)
        n_parameters = sum([p.numel() for p in self.model.parameters()])
        bs = self.dataloader.batch_size
        norm2 = 0
        for (inputs, targets) in self.dataloader:
            self._vTg = torch.zeros(inputs.size(0), device=device)
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            loss = self.loss_closure(inputs, targets)
            torch.autograd.grad(loss, [inputs])
            norm2 += (self._vTg**2).sum(dim=0)
        norm = (norm2 / n_examples) ** .5

        # remove hooks
        self.xs = dict()
        del self._vTg
        del self._v
        for h in self.handles:
            h.remove()

        return norm

    def implicit_trace(self):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex, self._hook_compute_trace)

        device = next(self.model.parameters()).device
        n_examples = len(self.dataloader.sampler)
        n_parameters = sum([p.numel() for p in self.model.parameters()])

        self._trace = 0
        for (inputs, targets) in self.dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            loss = self.loss_closure(inputs, targets)
            torch.autograd.grad(loss, [inputs])
        trace = self._trace / n_examples

        # remove hooks
        self.xs = dict()
        del self._trace
        for h in self.handles:
            h.remove()

        return trace

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
            self.grads[self.start:self.start+bs, start_p:start_p+mod.weight.numel()].add_(torch.bmm(gy.unsqueeze(2), x.unsqueeze(1)).view(bs, -1))
            if mod.bias is not None:
                self.grads[self.start:self.start+bs, start_p+mod.weight.numel():start_p+mod.weight.numel()+mod.bias.numel()] \
                    .add_(gy)
        elif mod_class == 'Conv2d':
            indiv_gw = per_example_grad_conv(mod, x, gy)
            self.grads[self.start:self.start+bs, start_p:start_p+mod.weight.numel()].add_(indiv_gw.view(bs, -1))
            if mod.bias is not None:
                self.grads[self.start:self.start+bs, start_p+mod.weight.numel():start_p+mod.weight.numel()+mod.bias.numel()] \
                    .add_(gy.sum(dim=(2,3)))
        else:
            raise NotImplementedError

    def _hook_compute_layer_blocks(self, mod, grad_input, grad_output):
        mod_class = mod.__class__.__name__
        gy = grad_output[0]
        x = self.xs[mod]
        bs = x.size(0)
        block = self._blocks[mod]
        if mod_class == 'Linear':
            gw = torch.bmm(gy.unsqueeze(2), x.unsqueeze(1)).view(bs, -1)
            if mod.bias is not None:
                gw = torch.cat([gw.view(bs, -1), gy.view(bs, -1)], dim=1)
            block.add_(torch.mm(gw.t(), gw))
        elif mod_class == 'Conv2d':
            gw = per_example_grad_conv(mod, x, gy)
            spatial_positions = gy.size(2) * gy.size(3)
            if mod.bias is not None:
                gw = torch.cat([gw.view(bs, -1), gy.sum(dim=(2, 3)).view(bs, -1)], dim=1)
            block.add_(torch.mm(gw.t(), gw))
        else:
            raise NotImplementedError

    def _hook_compute_kfac_blocks(self, mod, grad_input, grad_output):
        mod_class = mod.__class__.__name__
        gy = grad_output[0]
        x = self.xs[mod]
        bs = x.size(0)
        block = self._blocks[mod]
        if mod_class == 'Linear':
            block[1].add_(torch.mm(gy.t(), gy))
            if mod.bias is not None:
                x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
            block[0].add_(torch.mm(x.t(), x))
        else:
            raise NotImplementedError

    def _hook_compute_diag(self, mod, grad_input, grad_output):
        mod_class = mod.__class__.__name__
        gy = grad_output[0]
        x = self.xs[mod]
        bs = x.size(0)
        start_p = self.p_pos[mod]
        if mod_class == 'Linear':
            self.diag_m[start_p:start_p+mod.weight.numel()].add_(torch.mm(gy.t()**2, x**2).view(-1))
            if mod.bias is not None:
                self.diag_m[start_p+mod.weight.numel():start_p+mod.weight.numel()+mod.bias.numel()] \
                    .add_((gy**2).sum(dim=0))
        else:
            raise NotImplementedError

    def _hook_compute_m_norm(self, mod, grad_input, grad_output):
        mod_class = mod.__class__.__name__
        gy = grad_output[0]
        x = self.xs[mod]
        bs = x.size(0)
        if mod_class == 'Linear':
            self._vTg += (torch.mm(x, self._v[mod.weight].t()) * gy).sum(dim=1)
            if mod.bias is not None:
                self._vTg += torch.mv(gy, self._v[mod.bias])
        elif mod_class == 'Conv2d':
            gy2 = F.conv2d(x, self._v[mod.weight], stride=mod.stride, padding=mod.padding, dilation=mod.dilation)
            self._vTg += (gy * gy2).view(bs, -1).sum(dim=1)
            if mod.bias is not None:
                self._vTg += torch.mv(gy.sum(dim=(2, 3)), self._v[mod.bias])
        else:
            raise NotImplementedError

    def _hook_compute_trace(self, mod, grad_input, grad_output):
        mod_class = mod.__class__.__name__
        gy = grad_output[0]
        x = self.xs[mod]
        bs = x.size(0)
        if mod_class == 'Linear':
            self._trace += torch.mm(gy.t()**2, x**2).sum()
            if mod.bias is not None:
                self._trace += (gy**2).sum()
        elif mod_class == 'Conv2d':
            indiv_gw = per_example_grad_conv(mod, x, gy)
            self._trace += (indiv_gw**2).sum()
            if mod.bias is not None:
                self._trace += (gy.sum(dim=(2,3))**2).sum()
        else:
            raise NotImplementedError