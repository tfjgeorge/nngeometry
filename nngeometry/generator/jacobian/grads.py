import torch
from nngeometry.layercollection import (LinearLayer, Conv2dLayer)
from nngeometry.utils import per_example_grad_conv
import torch.nn.functional as F


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

    @classmethod
    def quasidiag(cls, buffer_diag, buffer_cross, mod, layer, x, gy):
        w_numel = layer.weight.numel()
        buffer_diag[:w_numel].add_(torch.mm(gy.t()**2, x**2).view(-1))
        if layer.bias is not None:
            buffer_diag[w_numel:].add_((gy**2).sum(dim=0))
            buffer_cross.add_(torch.mm(gy.t()**2, x))


class Conv2dJacobianFactory(JacobianFactory):
    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy, bs):
        w_numel = layer.weight.numel()
        indiv_gw = per_example_grad_conv(mod, x, gy)
        buffer[:, :w_numel].add_(indiv_gw.view(bs, -1))
        if layer.bias is not None:
            buffer[:, w_numel:].add_(gy.sum(dim=(2, 3)))
            
    @classmethod
    def Jv(cls, buffer, mod, layer, x, gy, bs, v, v_bias):
        gy2 = F.conv2d(x, v, stride=mod.stride,
                       padding=mod.padding, dilation=mod.dilation)
        buffer.add_((gy * gy2).view(bs, -1).sum(dim=1))
        if layer.bias is not None:
            buffer.add_(torch.mv(gy.sum(dim=(2, 3)), v_bias))
            
    @classmethod
    def kfac_xx(cls, buffer, mod, layer, x, gy):
        ks = (mod.weight.size(2), mod.weight.size(3))
        # A_tilda in KFC
        A_tilda = F.unfold(x, kernel_size=ks, stride=mod.stride,
                            padding=mod.padding, dilation=mod.dilation)
        # A_tilda is bs * #locations x #parameters
        A_tilda = A_tilda.permute(0, 2, 1).contiguous() \
            .view(-1, A_tilda.size(1))
        if layer.bias is not None:
            A_tilda = torch.cat([A_tilda,
                                    torch.ones_like(A_tilda[:, :1])],
                                dim=1)
        # Omega_hat in KFC
        buffer.add_(torch.mm(A_tilda.t(), A_tilda))

    @classmethod
    def kfac_gg(cls, buffer, mod, layer, x, gy):
        spatial_locations = gy.size(2) * gy.size(3)
        os = gy.size(1)
        # DS_tilda in KFC
        DS_tilda = gy.permute(0, 2, 3, 1).contiguous().view(-1, os)
        buffer.add_(torch.mm(DS_tilda.t(), DS_tilda) / spatial_locations)

    @classmethod
    def kfe_diag(cls, buffer, mod, layer, x, gy, evecs_a, evecs_g):
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
        buffer.add_((indiv_gw**2).sum(dim=0).view(-1))

    @classmethod
    def quasidiag(cls, buffer_diag, buffer_cross, mod, layer, x, gy):
        w_numel = layer.weight.numel()
        indiv_gw = per_example_grad_conv(mod, x, gy)
        buffer_diag[:w_numel].add_((indiv_gw**2).sum(dim=0).view(-1))
        if layer.bias is not None:
            gb_per_example = gy.sum(dim=(2, 3))
            buffer_diag[w_numel:].add_((gb_per_example**2).sum(dim=0))
            y = (gy * gb_per_example.unsqueeze(2).unsqueeze(3))
            cross_this = F.conv2d(x.transpose(0, 1),
                                y.transpose(0, 1),
                                stride=mod.dilation,
                                padding=mod.padding,
                                dilation=mod.stride).transpose(0, 1)
            cross_this = cross_this[:, :, :mod.kernel_size[0], :mod.kernel_size[1]]
            buffer_cross.add_(cross_this)

FactoryMap = {
    LinearLayer: LinearJacobianFactory,
    Conv2dLayer: Conv2dJacobianFactory,
}