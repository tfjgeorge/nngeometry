import torch
from nngeometry.layercollection import LinearLayer


class JacobianFactory:
    @classmethod
    def diag(cls, buffer, mod, layer, x, gy, bs):
        buffer_flat = torch.zeros(bs, layer.numel(), device=buffer.device)
        cls.flat_grad(buffer_flat, mod, layer, x, gy, bs)
        buffer.add_((buffer_flat**2).sum(dim=0))

    @classmethod
    def trace(cls, buffer, mod, layer, x, gy, bs):
        buffer_diag = torch.zeros(layer.numel(), device=buffer.device)
        cls.diag(buffer_diag, mod, layer, x, gy, bs)
        buffer.add_(buffer_diag.sum())

    @classmethod
    def layer_block(cls, buffer, mod, layer, x, gy, bs):
        buffer_flat = torch.zeros(bs, layer.numel(), device=buffer.device)
        cls.flat_grad(buffer_flat, mod, layer, x, gy, bs)
        buffer.add_(torch.mm(buffer_flat.t(), buffer_flat))

    @classmethod
    def kxy(cls, buffer, mod, layer, x_i, gy_i, bs_i, x_o, gy_o, bs_o):
        buffer_flat_i = torch.zeros(bs_i, layer.numel(), device=buffer.device)
        buffer_flat_o = torch.zeros(bs_o, layer.numel(), device=buffer.device)
        cls.flat_grad(buffer_flat_i, mod, layer, x_i, gy_i, bs_i)
        cls.flat_grad(buffer_flat_o, mod, layer, x_o, gy_o, bs_o)
        buffer.add_(torch.mm(buffer_flat_i, buffer_flat_o.t()))

    @classmethod
    def Jv(cls, buffer, mod, layer, x, gy, bs, v, v_bias):
        buffer_flat = torch.zeros(bs, layer.numel(), device=buffer.device)
        cls.flat_grad(buffer_flat, mod, layer, x, gy, bs)
        v = v.view(-1)
        if v_bias is not None:
            v = torch.cat((v, v_bias))
        buffer.add_(torch.mv(buffer_flat, v))


class LinearJacobianFactory(JacobianFactory):
    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy, bs):
        w_numel = layer.weight.numel()
        buffer[:, :w_numel] \
            .add_(torch.bmm(gy.unsqueeze(2), x.unsqueeze(1)).view(bs, -1))
        if layer.bias is not None:
            buffer[:, w_numel:].add_(gy)

    @classmethod
    def diag(cls, buffer, mod, layer, x, gy, bs):
        w_numel = layer.weight.numel()
        buffer[:w_numel].add_(torch.mm(gy.t()**2, x**2).view(-1))
        if layer.bias is not None:
            buffer[w_numel:].add_((gy**2).sum(dim=0))

    @classmethod
    def kxy(cls, buffer, mod, layer, x_i, gy_i, bs_i, x_o, gy_o, bs_o):
        buffer.add_(torch.mm(x_i, x_o.t()) *
                    torch.mm(gy_i, gy_o.t()))
        if layer.bias is not None:
            buffer.add_(torch.mm(gy_i, gy_o.t()))

    @classmethod
    def Jv(cls, buffer, mod, layer, x, gy, bs, v, v_bias):
        buffer.add_((torch.mm(x, v.t()) * gy).sum(dim=1))
        if layer.bias is not None:
            buffer.add_(torch.mv(gy.contiguous(), v_bias))

    @classmethod
    def trace(cls, buffer, mod, layer, x, gy, bs):
        buffer.add_(torch.mm(gy.t()**2, x**2).sum())
        if layer.bias is not None:
            buffer.add_((gy**2).sum())

    @classmethod
    def kfac_xx(cls, buffer, mod, layer, x, gy):
        if layer.bias is not None:
            x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        buffer.add_(torch.mm(x.t(), x))

    @classmethod
    def kfac_gg(cls, buffer, mod, layer, x, gy):
        buffer.add_(torch.mm(gy.t(), gy))

    @classmethod
    def kfe_diag(cls, buffer, mod, layer, x, gy, evecs_a, evecs_g):
        if layer.bias is not None:
            x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        gy_kfe = torch.mm(gy, evecs_g)
        x_kfe = torch.mm(x, evecs_a)
        buffer.add_(torch.mm(gy_kfe.t()**2, x_kfe**2).view(-1))



FactoryMap = {
    LinearLayer: LinearJacobianFactory
}