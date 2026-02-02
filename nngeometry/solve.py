def zero_like_pvector(pvector):
    zero_pvector = pvector.clone()
    zero_pvector *= 0
    return zero_pvector


def cg(A, b, regul=1e-8, x0=None, rtol=1e-5, atol=0, max_iter=None, M=None):
    tol = max(rtol * b.norm(), atol)
    if max_iter is None:
        max_iter = 10 * A.layer_collection.numel()

    if x0 is None:
        r = b
        x = zero_like_pvector(r)
    else:
        r = b - A.mv(x0)
        if regul > 0:  # should we have a PMatImplicitDamp ?
            r = r - regul * x0
        x = x0

    z = r if M is None else M.solve(r, regul=regul)
    p = z
    for _ in range(max_iter):
        if r.norm() <= tol:
            return x
        Ap = A.mv(p)
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
