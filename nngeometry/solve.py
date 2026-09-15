import torch


def cg(A, b, regul=1e-8, x0=None, rtol=1e-5, atol=0, max_iter=None, M=None):
    tol = max(rtol * b.norm(), atol)
    lc = A.layer_collection
    if max_iter is None:
        max_iter = 10 * lc.numel()

    if x0 is None:
        r = b
        x = 0 * b
    else:
        r = b - A @ x0
        if regul > 0:  # should we have a PMatImplicitDamp ?
            r = r - regul * x0
        x = x0

    z = r if M is None else M.solve(r, regul=regul)
    p = z
    for _ in range(max_iter):
        if r.norm() <= tol:
            break
        Ap = A @ p
        if regul > 0:
            Ap = Ap + regul * p
        rz = r @ z
        α = rz / (p @ Ap)
        x = x + α * p
        r = r - α * Ap
        z = r if M is None else M.solve(r, regul=regul)
        β = (r @ z) / rz
        p = z + β * p
    return x


def qr(pfmap):
    from nngeometry.object.fspace import FMatDense
    from nngeometry.object.map import PFMapDense

    sJ = pfmap.size()

    Q, R = torch.linalg.qr(pfmap.to_torch().view(-1, sJ[-1]).t(), mode="reduced")
    Q = PFMapDense(pfmap.layer_collection, pfmap.generator, data=Q.t().view(*sJ))
    R = FMatDense(
        pfmap.layer_collection, pfmap.generator, data=R.view(sJ[0], sJ[1], sJ[0], sJ[1])
    )
    return Q, R


def block_cg(A, b, regul=1e-8, x0=None, rtol=1e-5, atol=0, max_iter=None, M=None):
    # https://arxiv.org/pdf/2502.16998 Algorithm 8 with γ->α and δ->β to match cg
    tol = torch.clamp(rtol * (torch.sum(b.to_torch() ** 2, dim=-1) ** 0.5), min=atol)
    lc = A.layer_collection
    if max_iter is None:
        max_iter = 10 * lc.numel()

    if x0 is None:
        r = b
        x = 0 * b
    else:
        r = b - (A @ x0.adjoint()).adjoint()
        if regul > 0:  # should we have a PMatImplicitDamp ?
            r = r - regul * x0
        x = x0

    z = r if M is None else M.solve(r, regul=regul)
    p = z
    p, _ = qr(p)
    for i in range(max_iter):
        if torch.all(torch.sum(r.to_torch() ** 2, dim=-1) ** 0.5 <= tol):
            break
        Ap = (A @ p.adjoint()).adjoint()
        if regul > 0:
            Ap = Ap + regul * p
        pTAp = p @ Ap.adjoint()
        α = r @ pTAp.solve(p, regul=0).adjoint()
        x = x + α @ p
        r = r - α @ Ap
        z = r if M is None else M.solve(r, regul=regul)
        β = -1 * (z @ pTAp.solve(Ap, regul=0).adjoint())
        p = z + β @ p
        p, _ = qr(p)
    return x
