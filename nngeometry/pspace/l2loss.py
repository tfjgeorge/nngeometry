import torch
import torch.nn.functional as torchF

class L2Loss:
    def __init__(self, model, dataloader, loss_closure):
        self.model = model
        self.dataloader = dataloader
        self.handles = []
        self.xs = dict()
        self.p_pos = dict() # maps parameters to their position in flattened representation
        self.mods = self._get_individual_modules(model)
        self.loss_closure = loss_closure

    def get_n_parameters(self):
        return sum([p.numel() for p in self.model.parameters()])

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

    def _get_individual_modules(self, model):
        mods = []
        sizes_mods = []
        parameters = []
        start = 0
        for mod in model.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                mods.append(mod)
                self.p_pos[mod] = start
                sizes_mods.append(mod.weight.size())
                parameters.append(mod.weight)
                start += mod.weight.numel()
                if mod.bias is not None:
                    sizes_mods.append(mod.bias.size())
                    parameters.append(mod.bias)
                    start += mod.bias.numel()

        # check order of flattening
        sizes_flat = [p.size() for p in model.parameters() if p.requires_grad]
        assert sizes_mods == sizes_flat
        # check that all parameters were added
        # will fail if using exotic layers such as BatchNorm
        assert len(set(parameters) - set(model.parameters())) == 0
        return mods

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
            indiv_gw = self._per_example_grad_conv(mod, x, gy)
            self.grads[self.start:self.start+bs, start_p:start_p+mod.weight.numel()].add_(indiv_gw.view(bs, -1))
            if mod.bias is not None:
                self.grads[self.start:self.start+bs, start_p+mod.weight.numel():start_p+mod.weight.numel()+mod.bias.numel()] \
                    .add_(gy.sum(dim=(2,3)))
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
            indiv_gw = self._per_example_grad_conv(mod, x, gy)
            self._vTg += torch.mv(indiv_gw.view(bs, -1), self._v[mod.weight].view(-1))
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
            indiv_gw = self._per_example_grad_conv(mod, x, gy)
            self._trace += (indiv_gw**2).sum()
            if mod.bias is not None:
                self._trace += (gy.sum(dim=(2,3))**2).sum()
        else:
            raise NotImplementedError


    def _per_example_grad_conv(self, mod, x, gy):
            ks = (mod.weight.size(2), mod.weight.size(3))
            gy_s = gy.size()
            bs = gy_s[0]
            x_unfold = torchF.unfold(x, kernel_size=ks, stride=mod.stride, padding=mod.padding, dilation=mod.dilation)
            x_unfold_s = x_unfold.size()
            return torch.bmm(gy.view(bs, gy_s[1], -1), x_unfold.view(bs, x_unfold_s[1], -1).permute(0, 2, 1))
