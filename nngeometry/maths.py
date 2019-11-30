def kronecker(A, B):
    sA = A.size()
    sB = B.size()
    return (A.view(sA[0], 1, sA[1], 1) * B.view(1, sB[0], 1, sB[1])) \
        .contiguous().view(sA[0] * sB[0], sA[1] * sB[1])
