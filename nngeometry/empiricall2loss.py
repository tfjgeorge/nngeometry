import torch

class EmpiricalL2Loss:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.handles = []
        self.xs = dict()
        self.p_pos = dict() # maps parameters to their position in flattened representation
        self.mods = self._get_individual_modules(model)

    def get_matrix(self, loss_closure):
        # add hooks
        self.handles += self._add_hooks(self._hook_savex, self._hook_compute_flat_grad)

        device = next(self.model.parameters()).device
        n_examples = len(self.dataloader.sampler)
        n_parameters = sum([p.numel() for p in self.model.parameters()])
        bs = self.dataloader.batch_size
        G = torch.zeros((n_parameters, n_parameters), device=device)
        self.grads = torch.zeros((bs, n_parameters), device=device)
        for (inputs, targets) in self.dataloader:
            self.grads.zero_()
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            loss = loss_closure(inputs, targets)
            torch.autograd.grad(loss, [inputs])
            G += torch.mm(self.grads.t(), self.grads)
        G /= n_examples

        # remove hooks
        for h in self.handles:
            h.remove()

        return G

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

        #check order of flattening
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
        start = self.p_pos[mod]
        if mod_class == 'Linear':
            self.grads[:, start:start+mod.weight.numel()].add_(torch.bmm(gy.unsqueeze(2), x.unsqueeze(1)).view(bs, -1))
            if mod.bias is not None:
                self.grads[:, start+mod.weight.numel():start+mod.weight.numel()+mod.bias.numel()] \
                    .add_(gy)
        else:
            raise NotImplementedError