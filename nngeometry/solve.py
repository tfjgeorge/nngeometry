from copy import deepcopy

import torch

from nngeometry.object.fspace import FMatDense
from nngeometry.object.map import PFMapDense


def zero_like(x):
    zero_x = deepcopy(x)
    zero_x *= 0
    return zero_x


def cg(A, b, regul=1e-8, x0=None, rtol=1e-5, atol=0, max_iter=None, M=None):
    # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
    # with some defaults from scipy
    tol = max(rtol * b.norm(), atol)
    lc = A.layer_collection
    if max_iter is None:
        max_iter = 10 * lc.numel()

    if x0 is None:
        r = b
        x = zero_like(r)
    else:
        r = b - A @ x0
        if regul > 0:  # should we have a PMatImplicitDamp ?
            r = r - regul * x0
        x = x0

    z = r if M is None else M.solve(r, regul=regul)
    p = z
    for i in range(max_iter):
        if r.norm() <= tol:
            return x
        Ap = A @ p
        if regul > 0:
            Ap = Ap + regul * p
        rz = r.dot(z)
        α = rz / p.dot(Ap)
        x = x + α * p
        r = r - α * Ap
        z = r if M is None else M.solve(r, regul=regul)
        β = r.dot(z) / rz
        p = z + β * p
    return x


def fmat(j1, j2):
    sj1, sj2 = j1.size(), j2.size()
    data = torch.mm(
        j1.to_torch().view(-1, sj1[-1]), j2.to_torch().view(-1, sj2[-1]).T
    ).view(sj1[0], sj1[1], sj2[0], sj2[1])
    return FMatDense(j1.layer_collection, j1.generator, data=data)


def solve_fmat(f1, f2, regul=0):
    sf1, sf2 = f1.size(), f2.size()
    f1_torch = f1.to_torch().view(sf1[0] * sf1[1], sf1[2] * sf1[3])
    if regul > 0:
        f1_torch += regul * torch.eye(sf1[0] * sf1[1])
    return FMatDense(
        f1.layer_collection,
        generator=f1.generator,
        data=torch.linalg.solve(
            f1_torch,
            f2.to_torch().view(sf2[0] * sf2[1], sf2[2] * sf2[3]),
        ).view(sf1[0], sf1[1], sf2[2], sf2[3]),
    )


def Q(pfmap):
    return PFMapDense(
        pfmap.layer_collection,
        generator=pfmap.generator,
        data=torch.linalg.qr(
            pfmap.to_torch().view(-1, pfmap.layer_collection.numel()).T
        )[0].T.view(*pfmap.size()),
    )


def block_cg(A, b, regul=1e-8, x0=None, rtol=1e-5, atol=0, max_iter=None, M=None):
    # https://arxiv.org/pdf/2502.16998 Algorithm 6
    tol = max(rtol * torch.sum(b.to_torch() ** 2, dim=-1).mean(), atol)
    lc = A.layer_collection
    if max_iter is None:
        max_iter = 10 * lc.numel()

    if x0 is None:
        r = b
        x = zero_like(r)
    else:
        r = b - A @ x0
        if regul > 0:  # should we have a PMatImplicitDamp ?
            r = r - regul * x0
        x = x0

    z = r if M is None else M.solve(r, regul=regul)
    p = z
    p = Q(p)
    for i in range(max_iter):
        if torch.all(torch.sum(r.to_torch() ** 2, dim=-1) <= tol):
            return x
        Ap = A @ p
        if regul > 0:
            Ap = Ap + regul * p
        rz = fmat(p, z)
        pTAp = fmat(p, Ap)
        α = solve_fmat(pTAp, rz, regul=0)
        x = x + p @ α
        r = r - Ap @ α
        z = r if M is None else M.solve(r)
        β = solve_fmat(pTAp, fmat(z, Ap).T)
        β *= -1
        p = z + p @ β
        p = Q(p)
    return x
