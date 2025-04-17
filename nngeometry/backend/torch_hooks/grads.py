import torch
import torch.nn.functional as F

from nngeometry.layercollection import (
    Affine1dLayer,
    BatchNorm1dLayer,
    BatchNorm2dLayer,
    Conv1dLayer,
    Conv2dLayer,
    ConvTranspose2dLayer,
    Cosine1dLayer,
    EmbeddingLayer,
    GroupNormLayer,
    LayerNormLayer,
    LinearLayer,
    WeightNorm1dLayer,
    WeightNorm2dLayer,
)

from .grads_conv import conv1d_backward, conv2d_backward, convtranspose2d_backward


class JacobianFactory:
    @classmethod
    def diag(cls, buffer, mod, layer, x, gy):
        bs = x.size(0)
        buffer_flat = torch.zeros(bs, layer.numel(), device=buffer.device)
        cls.flat_grad(buffer_flat, mod, layer, x, gy)
        buffer.add_((buffer_flat**2).sum(dim=0))

    @classmethod
    def trace(cls, buffer, mod, layer, x, gy):
        buffer_diag = torch.zeros(layer.numel(), device=buffer.device)
        cls.diag(buffer_diag, mod, layer, x, gy)
        buffer.add_(buffer_diag.sum())

    @classmethod
    def layer_block(cls, buffer, mod, layer, x, gy):
        bs = x.size(0)
        buffer_flat = torch.zeros(bs, layer.numel(), device=buffer.device)
        cls.flat_grad(buffer_flat, mod, layer, x, gy)
        buffer.add_(torch.mm(buffer_flat.t(), buffer_flat))

    @classmethod
    def kxy(cls, buffer, mod, layer, x_i, gy_i, x_o, gy_o):
        bs_i = x_i.size(0)
        bs_o = x_o.size(0)
        buffer_flat_i = torch.zeros(bs_i, layer.numel(), device=buffer.device)
        buffer_flat_o = torch.zeros(bs_o, layer.numel(), device=buffer.device)
        cls.flat_grad(buffer_flat_i, mod, layer, x_i, gy_i)
        cls.flat_grad(buffer_flat_o, mod, layer, x_o, gy_o)
        buffer.add_(torch.mm(buffer_flat_i, buffer_flat_o.t()))

    @classmethod
    def Jv(cls, buffer, mod, layer, x, gy, v, v_bias):
        bs = x.size(0)
        buffer_flat = torch.zeros(bs, layer.numel(), device=buffer.device)
        cls.flat_grad(buffer_flat, mod, layer, x, gy)
        v = v.view(-1)
        if v_bias is not None:
            v_bias = v_bias.view(-1)
            v = torch.cat((v, v_bias))
        buffer.add_(torch.mv(buffer_flat, v))


class LinearJacobianFactory(JacobianFactory):
    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy):
        bs = x.size(0)
        w_numel = layer.weight.numel()
        if gy.ndim == 2:
            gy = gy[:, None, :]
            x = x[:, None, :]
        buffer[:, :w_numel].add_(torch.bmm(gy.transpose(1, 2), x).view(bs, -1))
        if layer.bias is not None:
            buffer[:, w_numel:].add_(gy.sum(dim=1))

    @classmethod
    def diag(cls, buffer, mod, layer, x, gy):
        if gy.ndim > 2:
            return super(LinearJacobianFactory, cls).diag(buffer, mod, layer, x, gy)

        w_numel = layer.weight.numel()
        buffer[:w_numel].add_(torch.mm(gy.t() ** 2, x**2).view(-1))
        if layer.bias is not None:
            buffer[w_numel:].add_((gy**2).sum(dim=0))

    @classmethod
    def kxy(cls, buffer, mod, layer, x_i, gy_i, x_o, gy_o):
        if gy_i.ndim > 2:
            return super(LinearJacobianFactory, cls).kxy(
                buffer, mod, layer, x_i, gy_i, x_o, gy_o
            )
        buffer.add_(torch.mm(x_i, x_o.t()) * torch.mm(gy_i, gy_o.t()))
        if layer.bias is not None:
            buffer.add_(torch.mm(gy_i, gy_o.t()))

    @classmethod
    def Jv(cls, buffer, mod, layer, x, gy, v, v_bias):
        x_s = x.size()
        gy_s = gy.size()
        buffer.add_(
            (torch.mm(x.reshape(-1, x_s[-1]), v.t()) * gy.reshape(-1, gy_s[-1]))
            .reshape(x_s[0], -1)
            .sum(dim=1)
        )

        if layer.bias is not None:
            if gy.ndim == 2:
                buffer.add_(torch.mv(gy, v_bias))
            elif gy.ndim == 3:
                buffer.add_(
                    torch.mv(gy.reshape(-1, gy_s[-1]), v_bias)
                    .reshape(gy_s[0], gy_s[1])
                    .sum(dim=1)
                )

    @classmethod
    def trace(cls, buffer, mod, layer, x, gy):
        if gy.ndim > 2:
            return super(LinearJacobianFactory, cls).trace(buffer, mod, layer, x, gy)

        buffer.add_(torch.mm(gy.t() ** 2, x**2).sum())
        if layer.bias is not None:
            buffer.add_((gy**2).sum())

    @classmethod
    def kfac_xx(cls, buffer, mod, layer, x, gy):
        if x.ndim == 3:
            x = x.reshape(-1, x.size(-1))
        if layer.bias is not None:
            x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        buffer.add_(torch.mm(x.t(), x))

    @classmethod
    def kfac_gg(cls, buffer, mod, layer, x, gy):
        spatial_locations = 1
        if gy.ndim == 3:
            spatial_locations = gy.size(1)
            gy = gy.reshape(-1, gy.size(-1))

        buffer.add_(torch.mm(gy.t(), gy)) / spatial_locations

    @classmethod
    def kfe_diag(cls, buffer, mod, layer, x, gy, evecs_a, evecs_g):
        x_s = x.size()
        gy_s = gy.size()
        if gy.ndim == 2:
            if layer.bias is not None:
                x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
            gy_kfe = torch.mm(gy, evecs_g)
            x_kfe = torch.mm(x, evecs_a)
            buffer.add_(torch.mm(gy_kfe.t() ** 2, x_kfe**2).view(-1))
        elif gy.ndim == 3:
            if layer.bias is not None:
                x = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=2)
            gy_kfe = torch.mm(gy.view(-1, gy_s[2]), evecs_g)
            x_kfe = torch.mm(x.view(-1, evecs_a.size(0)), evecs_a)

            per_ex_kfe_grad = torch.bmm(
                gy_kfe.view(*gy_s).transpose(1, 2), x_kfe.view(x_s[0], x_s[1], -1)
            )
            buffer.add_((per_ex_kfe_grad**2).sum(dim=0).view(-1))

    @classmethod
    def quasidiag(cls, buffer_diag, buffer_cross, mod, layer, x, gy):
        w_numel = layer.weight.numel()
        buffer_diag[:w_numel].add_(torch.mm(gy.t() ** 2, x**2).view(-1))
        if layer.bias is not None:
            buffer_diag[w_numel:].add_((gy**2).sum(dim=0))
            buffer_cross.add_(torch.mm(gy.t() ** 2, x))


class Conv2dJacobianFactory(JacobianFactory):
    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy):
        bs = x.size(0)
        w_numel = layer.weight.numel()
        indiv_gw = conv2d_backward(mod, x, gy)
        buffer[:, :w_numel].add_(indiv_gw.view(bs, -1))
        if layer.bias is not None:
            buffer[:, w_numel:].add_(gy.sum(dim=(2, 3)))

    @classmethod
    def Jv(cls, buffer, mod, layer, x, gy, v, v_bias):
        bs = x.size(0)
        gy2 = F.conv2d(
            x, v, stride=mod.stride, padding=mod.padding, dilation=mod.dilation
        )
        buffer.add_((gy * gy2).view(bs, -1).sum(dim=1))
        if layer.bias is not None:
            buffer.add_(torch.mv(gy.sum(dim=(2, 3)), v_bias))

    @classmethod
    def kfac_xx(cls, buffer, mod, layer, x, gy):
        ks = (mod.weight.size(2), mod.weight.size(3))
        # A_tilda in KFC
        A_tilda = F.unfold(
            x,
            kernel_size=ks,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
        )
        # A_tilda is bs * #locations x #parameters
        A_tilda = A_tilda.permute(0, 2, 1).contiguous().view(-1, A_tilda.size(1))
        if layer.bias is not None:
            A_tilda = torch.cat([A_tilda, torch.ones_like(A_tilda[:, :1])], dim=1)
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
        x_unfold = F.unfold(
            x,
            kernel_size=ks,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
        )
        x_unfold_s = x_unfold.size()
        x_unfold = (
            x_unfold.view(bs, x_unfold_s[1], -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(-1, x_unfold_s[1])
        )
        if mod.bias is not None:
            x_unfold = torch.cat([x_unfold, torch.ones_like(x_unfold[:, :1])], dim=1)
        x_kfe = torch.mm(x_unfold, evecs_a)

        # project gy to kfe
        gy = gy.view(bs, gy_s[1], -1).permute(0, 2, 1).contiguous()
        gy_kfe = torch.mm(gy.view(-1, gy_s[1]), evecs_g)
        gy_kfe = gy_kfe.view(bs, -1, gy_s[1]).permute(0, 2, 1).contiguous()

        indiv_gw = torch.bmm(
            gy_kfe.view(bs, gy_s[1], -1), x_kfe.view(bs, -1, x_kfe.size(1))
        )
        buffer.add_((indiv_gw**2).sum(dim=0).view(-1))

    @classmethod
    def quasidiag(cls, buffer_diag, buffer_cross, mod, layer, x, gy):
        w_numel = layer.weight.numel()
        indiv_gw = conv2d_backward(mod, x, gy)
        buffer_diag[:w_numel].add_((indiv_gw**2).sum(dim=0).view(-1))
        if layer.bias is not None:
            gb_per_example = gy.sum(dim=(2, 3))
            buffer_diag[w_numel:].add_((gb_per_example**2).sum(dim=0))
            y = gy * gb_per_example.unsqueeze(2).unsqueeze(3)
            cross_this = F.conv2d(
                x.transpose(0, 1),
                y.transpose(0, 1),
                stride=mod.dilation,
                padding=mod.padding,
                dilation=mod.stride,
            ).transpose(0, 1)
            cross_this = cross_this[:, :, : mod.kernel_size[0], : mod.kernel_size[1]]
            buffer_cross.add_(cross_this)


class ConvTranspose2dJacobianFactory(JacobianFactory):
    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy):
        bs = x.size(0)
        w_numel = layer.weight.numel()
        indiv_gw = convtranspose2d_backward(mod, x, gy)
        buffer[:, :w_numel].add_(indiv_gw.view(bs, -1))
        if layer.bias is not None:
            buffer[:, w_numel:].add_(gy.sum(dim=(2, 3)))


def check_bn_training(mod):
    # check that BN layers are in eval mode
    if mod.training:
        raise NotImplementedError(
            "NNGeometry's Torch Hook backend can"
            + " only handle BatchNorm in evaluation mode"
        )


class BatchNorm1dJacobianFactory(JacobianFactory):
    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy):
        check_bn_training(mod)
        w_numel = layer.weight.numel()
        x_normalized = F.batch_norm(
            x, mod.running_mean, mod.running_var, None, None, mod.training, momentum=0.0
        )
        buffer[:, :w_numel].add_(gy * x_normalized)
        if layer.bias is not None:
            buffer[:, w_numel:].add_(gy)


class BatchNorm2dJacobianFactory(JacobianFactory):
    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy):
        check_bn_training(mod)
        w_numel = layer.weight.numel()
        x_normalized = F.batch_norm(
            x, mod.running_mean, mod.running_var, None, None, mod.training, momentum=0.0
        )
        buffer[:, :w_numel].add_((gy * x_normalized).sum(dim=(2, 3)))
        if layer.bias is not None:
            buffer[:, w_numel:].add_(gy.sum(dim=(2, 3)))


class LayerNormJacobianFactory(JacobianFactory):
    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy):
        w_numel = layer.weight.numel()
        bs = x.size(0)
        x_normalized = F.layer_norm(
            x, normalized_shape=mod.normalized_shape, eps=mod.eps
        )

        gy = gy.view(bs, -1, *mod.normalized_shape)
        x_normalized = x_normalized.view(bs, -1, *mod.normalized_shape)

        buffer[:, :w_numel].add_((gy * x_normalized).sum(dim=1).view(bs, -1))
        if layer.bias is not None:
            buffer[:, w_numel:].add_(gy.sum(dim=1).view(bs, -1))


class GroupNormJacobianFactory(JacobianFactory):
    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy):
        w_numel = layer.weight.numel()
        x_normalized = F.group_norm(x, mod.num_groups, eps=mod.eps)
        buffer[:, :w_numel].add_((gy * x_normalized).sum(dim=(2, 3)))
        buffer[:, w_numel:].add_(gy.sum(dim=(2, 3)))


class WeightNorm1dJacobianFactory(JacobianFactory):
    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy):
        bs = x.size(0)
        gw_prime = (
            torch.bmm(gy.unsqueeze(2), x.unsqueeze(1))
            .view(bs, -1)
            .view(bs, *mod.weight.size())
        )
        norm2 = (mod.weight**2).sum(dim=1, keepdim=True) + mod.eps

        gw = gw_prime / torch.sqrt(norm2).unsqueeze(0)

        gw -= (gw_prime * mod.weight.unsqueeze(0)).sum(dim=2, keepdim=True) * (
            mod.weight * norm2 ** (-1.5)
        ).unsqueeze(0)

        buffer.add_(gw.view(bs, -1))


class WeightNorm2dJacobianFactory(JacobianFactory):
    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy):
        bs = x.size(0)
        gw_prime = conv2d_backward(mod, x, gy).view(bs, *mod.weight.size())
        norm2 = (mod.weight**2).sum(dim=(1, 2, 3), keepdim=True) + mod.eps

        gw = gw_prime / torch.sqrt(norm2).unsqueeze(0)

        gw -= (gw_prime * mod.weight.unsqueeze(0)).sum(dim=(2, 3, 4), keepdim=True) * (
            mod.weight * norm2 ** (-1.5)
        ).unsqueeze(0)

        buffer.add_(gw.view(bs, -1))

    @classmethod
    def flat_grad_(cls, buffer, mod, layer, x, gy):
        bs = x.size(0)
        out_dim = mod.weight.size(0)
        norm2 = (mod.weight**2).sum(dim=(1, 2, 3)) + mod.eps
        gw = conv2d_backward(mod, x, gy / torch.sqrt(norm2).view(1, out_dim, 1, 1))
        gw = gw.view(bs, out_dim, -1)
        wn2_out = F.conv2d(
            x,
            mod.weight / norm2.view(out_dim, 1, 1, 1) ** 1.5,
            None,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
        )
        gw -= (gy * wn2_out).sum(dim=(2, 3)).view(bs, out_dim, 1) * mod.weight.view(
            1, out_dim, -1
        )
        buffer.add_(gw.view(bs, -1))


class Cosine1dJacobianFactory(JacobianFactory):
    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy):
        bs = x.size(0)
        norm2_w = (mod.weight**2).sum(dim=1, keepdim=True) + mod.eps
        norm2_x = (x**2).sum(dim=1, keepdim=True) + mod.eps
        x = x / torch.sqrt(norm2_x)
        gw = torch.bmm(gy.unsqueeze(2) / torch.sqrt(norm2_w), x.unsqueeze(1))
        wn2_out = F.linear(x, mod.weight / norm2_w**1.5)
        gw -= (gy * wn2_out).unsqueeze(2) * mod.weight.unsqueeze(0)
        buffer.add_(gw.view(bs, -1))


class Affine1dJacobianFactory(JacobianFactory):
    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy):
        w_numel = layer.weight.numel()
        buffer[:, :w_numel].add_(gy * x)
        if layer.bias is not None:
            buffer[:, w_numel:].add_(gy)


class Conv1dJacobianFactory(JacobianFactory):
    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy):
        bs = x.size(0)
        w_numel = layer.weight.numel()
        indiv_gw = conv1d_backward(mod, x, gy)
        buffer[:, :w_numel].add_(indiv_gw.view(bs, -1))
        if layer.bias is not None:
            buffer[:, w_numel:].add_(gy.sum(dim=2))

    @classmethod
    def Jv(cls, buffer, mod, layer, x, gy, v, v_bias):
        bs = x.size(0)
        gy2 = F.conv1d(
            x, v, stride=mod.stride, padding=mod.padding, dilation=mod.dilation
        )
        buffer.add_((gy * gy2).view(bs, -1).sum(dim=1))
        if layer.bias is not None:
            buffer.add_(torch.mv(gy.sum(dim=2), v_bias))

    @classmethod
    def kfac_xx(cls, buffer, mod, layer, x, gy):
        ks = (1, mod.weight.size(2))
        # A_tilda in KFC
        A_tilda = F.unfold(
            x.unsqueeze(2),
            kernel_size=ks,
            stride=(1, mod.stride[0]),
            padding=(0, mod.padding[0]),
            dilation=(1, mod.dilation[0]),
        )
        # A_tilda is bs * #locations x #parameters
        A_tilda = A_tilda.permute(0, 2, 1).contiguous().view(-1, A_tilda.size(1))
        if layer.bias is not None:
            A_tilda = torch.cat([A_tilda, torch.ones_like(A_tilda[:, :1])], dim=1)
        # Omega_hat in KFC
        buffer.add_(torch.mm(A_tilda.t(), A_tilda))

    @classmethod
    def kfac_gg(cls, buffer, mod, layer, x, gy):
        spatial_locations = gy.size(2)
        os = gy.size(1)
        # DS_tilda in KFC
        DS_tilda = gy.permute(0, 2, 1).contiguous().view(-1, os)
        buffer.add_(torch.mm(DS_tilda.t(), DS_tilda) / spatial_locations)

    @classmethod
    def kfe_diag(cls, buffer, mod, layer, x, gy, evecs_a, evecs_g):
        ks = (1, mod.weight.size(2))
        gy_s = gy.size()
        bs = gy_s[0]
        # project x to kfe
        x_unfold = F.unfold(
            x.unsqueeze(2),
            kernel_size=ks,
            stride=(1, mod.stride[0]),
            padding=(0, mod.padding[0]),
            dilation=(1, mod.dilation[0]),
        )
        x_unfold_s = x_unfold.size()
        x_unfold = (
            x_unfold.view(bs, x_unfold_s[1], -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(-1, x_unfold_s[1])
        )
        if mod.bias is not None:
            x_unfold = torch.cat([x_unfold, torch.ones_like(x_unfold[:, :1])], dim=1)
        x_kfe = torch.mm(x_unfold, evecs_a)

        # project gy to kfe
        gy = gy.view(bs, gy_s[1], -1).permute(0, 2, 1).contiguous()
        gy_kfe = torch.mm(gy.view(-1, gy_s[1]), evecs_g)
        gy_kfe = gy_kfe.view(bs, -1, gy_s[1]).permute(0, 2, 1).contiguous()

        indiv_gw = torch.bmm(
            gy_kfe.view(bs, gy_s[1], -1), x_kfe.view(bs, -1, x_kfe.size(1))
        )
        buffer.add_((indiv_gw**2).sum(dim=0).view(-1))


def check_embedding_arguments(mod):
    # check that embedding layers are set up with supported arguments
    if mod.max_norm is not None or mod.scale_grad_by_freq or mod.sparse:
        raise NotImplementedError(
            """NNGeometry's Torch Hook backend can currently only
            handle Embedding layers with default arguments"""
        )


class EmbeddingJacobianFactory(JacobianFactory):

    @classmethod
    def flat_grad(cls, buffer, mod, layer, x, gy):
        check_embedding_arguments(mod)
        x_s = x.size()
        x_onehot = F.one_hot(x, num_classes=layer.num_embeddings)
        w_numel = layer.weight.numel()
        buffer[:, :w_numel].add_(
            torch.bmm(x_onehot.transpose(1, 2).to(gy.dtype), gy).view(x_s[0], -1)
        )

    @classmethod
    def kfac_gg(cls, buffer, mod, layer, x, gy):
        # this uses the same suming and scaling as KFC
        spatial_locations = gy.size(1)
        os = gy.size(2)
        # DS_tilda in KFC
        gy = gy.view(-1, os)
        buffer.add_(torch.mm(gy.t(), gy) / spatial_locations)

    @classmethod
    def kfac_xx(cls, buffer, mod, layer, x, gy):
        x_s = x.size()
        x_onehot = F.one_hot(x, num_classes=layer.num_embeddings).reshape(
            x_s[0] * x_s[1], -1
        )
        buffer.add_(torch.mm(x_onehot.t(), x_onehot).to(buffer.dtype))

    @classmethod
    def kfe_diag(cls, buffer, mod, layer, x, gy, evecs_a, evecs_g):
        # x is bs * spatial
        # gy is bs * spatial * embedding_dim
        gy_s = gy.size()
        x_s = x.size()

        # project x to kfe
        x_onehot = F.one_hot(x, num_classes=layer.num_embeddings).reshape(
            x_s[0] * x_s[1], -1
        )
        x_kfe = torch.mm(x_onehot.to(evecs_a.dtype), evecs_a).view(x_s[0], x_s[1], -1)

        # project gy to kfe
        gy = gy.view(-1, gy_s[2])
        gy_kfe = torch.mm(gy, evecs_g).view(*gy_s)

        # per example gradients in KFE
        indiv_gw = torch.bmm(x_kfe.transpose(1, 2), gy_kfe).view(x_s[0], -1)

        buffer.add_((indiv_gw**2).sum(dim=0).view(-1))


FactoryMap = {
    LinearLayer: LinearJacobianFactory,
    Conv1dLayer: Conv1dJacobianFactory,
    Conv2dLayer: Conv2dJacobianFactory,
    ConvTranspose2dLayer: ConvTranspose2dJacobianFactory,
    BatchNorm1dLayer: BatchNorm1dJacobianFactory,
    BatchNorm2dLayer: BatchNorm2dJacobianFactory,
    GroupNormLayer: GroupNormJacobianFactory,
    WeightNorm1dLayer: WeightNorm1dJacobianFactory,
    WeightNorm2dLayer: WeightNorm2dJacobianFactory,
    Cosine1dLayer: Cosine1dJacobianFactory,
    Affine1dLayer: Affine1dJacobianFactory,
    LayerNormLayer: LayerNormJacobianFactory,
    LayerNormLayer: LayerNormJacobianFactory,
    EmbeddingLayer: EmbeddingJacobianFactory,
}
