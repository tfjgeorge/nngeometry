from nngeometry.layers import Cosine
import torch
from utils import check_ratio

def test_cosine():
    cosine_layer = Cosine(2, 3)

    # extract vector parrallel to cosine_layer.weight[0, :]
    x_0 = cosine_layer.weight[0, :]
    x_col = 3 * x_0
    # extract vector orthogonal to cosine_layer.weight[1, :]
    x_1 = cosine_layer.weight[1, :]
    x_2 = cosine_layer.weight[2, :]
    x_orth = x_2 - (x_2 * x_1).sum() / torch.norm(x_1)**2 * x_1
    x = torch.stack((x_col, x_orth))
    
    out = cosine_layer(x)

    check_ratio(1 + cosine_layer.bias[0], out[0, 0])
    check_ratio(cosine_layer.bias[1], out[1, 1])