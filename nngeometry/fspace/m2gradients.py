import torch
from ..utils import get_individual_modules, per_example_grad_conv, get_n_parameters

class M2Gradients:
    def __init__(self, model, dataloader, loss_function):
        self.model = model
        self.dataloader = dataloader
        self.handles = []
        self.x_outer = dict()
        self.x_inner = dict()
        self.xs = dict()
        self.gy_outer = dict()
        # self.p_pos maps parameters to their position in flattened representation
        self.mods, self.p_pos = get_individual_modules(model)
        self.loss_function = loss_function

    def release_buffers(self):
        self.x_outer = dict()
        self.x_inner = dict()
        self.xs = dict()
        self.gy_outer = dict()

    def get_matrix(self):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex_io, self._hook_kxy)

        device = next(self.model.parameters()).device
        n_examples = len(self.dataloader.sampler)
        n_parameters = sum([p.numel() for p in self.model.parameters()])
        bs = self.dataloader.batch_size
        self.G = torch.zeros((n_examples, n_examples), device=device)
        self.e_outer = 0
        for i_outer, (inputs, targets) in enumerate(self.dataloader):
            self.outerloop_switch = True # used in hooks to switch between store/compute
            inputs, targets = inputs.to(device), targets.to(device)
            bs_outer = targets.size(0)
            inputs.requires_grad = True
            loss = self.loss_function(inputs, targets).sum()
            torch.autograd.grad(loss, [inputs])
            self.outerloop_switch = False 

            self.e_inner = 0
            for i_inner, (inputs, targets) in enumerate(self.dataloader):
                if i_inner > i_outer:
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                inputs.requires_grad = True
                loss = self.loss_function(inputs, targets).sum()
                torch.autograd.grad(loss, [inputs])
                if i_inner < i_outer: # exclude diagonal
                    bs_inner = targets.size(0)
                    self.G[self.e_outer:self.e_outer+bs_outer, self.e_inner:self.e_inner+bs_inner] += \
                        self.G[self.e_inner:self.e_inner+bs_inner, self.e_outer:self.e_outer+bs_outer].t()
                self.e_inner += inputs.size(0)

            self.e_outer += inputs.size(0)
        G = self.G

        # remove hooks
        del self.e_inner, self.e_outer
        del self.G
        self.x_inner = dict()
        self.x_outer = dict()
        self.gy_outer = dict()
        for h in self.handles:
            h.remove()

        return G

    def implicit_vTMv(self, v):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex, self._hook_compute_cTv)

        device = next(self.model.parameters()).device
        n_examples = len(self.dataloader.sampler)
        n_parameters = get_n_parameters(self.model)
        bs = self.dataloader.batch_size

        self._cTv = torch.zeros((n_parameters,), device=device)
        i = 0
        for i_outer, (inputs, targets) in enumerate(self.dataloader):
            self._c = v[i:i+bs]
            i += bs
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            loss = self.loss_function(inputs, targets).sum()
            torch.autograd.grad(loss, [inputs])
        m_norm = (self._cTv**2).sum()

        # remove hooks
        del self._cTv
        self.xs = dict()
        for h in self.handles:
            h.remove()

        return m_norm

    def implicit_frobenius(self):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex_io, self._hook_compute_frobenius)

        device = next(self.model.parameters()).device
        n_examples = len(self.dataloader.sampler)
        n_parameters = get_n_parameters(self.model)
        bs = self.dataloader.batch_size
        norm2 = 0
        for i_outer, (inputs, targets) in enumerate(self.dataloader):
            self.outerloop_switch = True # used in hooks to switch between store/compute
            inputs, targets = inputs.to(device), targets.to(device)
            bs_outer = targets.size(0)
            inputs.requires_grad = True
            loss = self.loss_function(inputs, targets).sum()
            torch.autograd.grad(loss, [inputs])
            self.outerloop_switch = False 

            for i_inner, (inputs, targets) in enumerate(self.dataloader):
                if i_inner > i_outer:
                    break
                self.mb_block = torch.zeros((bs, bs), device=device)
                inputs, targets = inputs.to(device), targets.to(device)
                inputs.requires_grad = True
                loss = self.loss_function(inputs, targets).sum()
                torch.autograd.grad(loss, [inputs])
                this_mb_norm2 = (self.mb_block**2).sum()
                if i_inner < i_outer:
                    this_mb_norm2 *= 2
                norm2 += this_mb_norm2

        # remove hooks
        del self.mb_block
        self.x_inner = dict()
        self.x_outer = dict()
        for h in self.handles:
            h.remove()

        return norm2**.5

    def _add_hooks(self, hook_x, hook_gy):
        handles = []
        for m in self.mods:
            handles.append(m.register_forward_pre_hook(hook_x))
            handles.append(m.register_backward_hook(hook_gy))
        return handles

    def _hook_savex_io(self, mod, i):
        if self.outerloop_switch:
            self.x_outer[mod] = i[0]
        else:
            self.x_inner[mod] = i[0]

    def _hook_savex(self, mod, i):
        self.xs[mod] = i[0]

    def _hook_kxy(self, mod, grad_input, grad_output):
        if self.outerloop_switch:
            self.gy_outer[mod] = grad_output[0]
        else:
            mod_class = mod.__class__.__name__
            gy_inner = grad_output[0]
            gy_outer = self.gy_outer[mod]
            x_outer = self.x_outer[mod]
            x_inner = self.x_inner[mod]
            bs_inner = x_inner.size(0)
            bs_outer = x_outer.size(0)
            if mod_class == 'Linear':
                self.G[self.e_inner:self.e_inner+bs_inner, self.e_outer:self.e_outer+bs_outer] += \
                        torch.mm(x_inner, x_outer.t()) * torch.mm(gy_inner, gy_outer.t())
                if mod.bias is not None:
                    self.G[self.e_inner:self.e_inner+bs_inner, self.e_outer:self.e_outer+bs_outer] += \
                            torch.mm(gy_inner, gy_outer.t())
            elif mod_class == 'Conv2d':
                indiv_gw_inner = per_example_grad_conv(mod, x_inner, gy_inner).view(bs_inner, -1)
                indiv_gw_outer = per_example_grad_conv(mod, x_outer, gy_outer).view(bs_outer, -1)
                self.G[self.e_inner:self.e_inner+bs_inner, self.e_outer:self.e_outer+bs_outer] += \
                        torch.mm(indiv_gw_inner, indiv_gw_outer.t())
                if mod.bias is not None:
                    self.G[self.e_inner:self.e_inner+bs_inner, self.e_outer:self.e_outer+bs_outer] += \
                            torch.mm(gy_inner.sum(dim=(2,3)), gy_outer.sum(dim=(2,3)).t())
            else:
                raise NotImplementedError

    def _hook_compute_frobenius(self, mod, grad_input, grad_output):
        if self.outerloop_switch:
            self.gy_outer[mod] = grad_output[0]
        else:
            mod_class = mod.__class__.__name__
            gy_inner = grad_output[0]
            gy_outer = self.gy_outer[mod]
            x_outer = self.x_outer[mod]
            x_inner = self.x_inner[mod]
            bs_inner = x_inner.size(0)
            bs_outer = x_outer.size(0)
            if mod_class == 'Linear':
                self.mb_block[:bs_inner, :bs_outer] += \
                        torch.mm(x_inner, x_outer.t()) * torch.mm(gy_inner, gy_outer.t())
                if mod.bias is not None:
                    self.mb_block[:bs_inner, :bs_outer] += torch.mm(gy_inner, gy_outer.t())
            else:
                raise NotImplementedError

    def _hook_compute_cTv(self, mod, grad_input, grad_output):
        mod_class = mod.__class__.__name__
        gy = grad_output[0]
        xs = self.xs[mod]
        start = self.p_pos[mod]
        if mod_class == 'Linear':
            self._cTv[start:start+mod.weight.numel()] += torch.mm(gy.t(), self._c.view(-1, 1) * xs).view(-1)
            if mod.bias is not None:
                self._cTv[start+mod.weight.numel():start+mod.weight.numel()+mod.bias.numel()] += \
                    torch.mv(gy.t(), self._c)
        else:
            raise NotImplementedError
