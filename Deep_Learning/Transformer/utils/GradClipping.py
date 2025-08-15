import torch as T
from torch import nn

def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = T.sqrt(sum(T.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

if __name__ == "__main__":
    """测试梯度裁剪"""
    import torch.nn as nn
    
    net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 1))
    for param in net.parameters():
        param.grad = T.ones_like(param)
    grad_clipping(net, 1)
    print([param.grad.norm() for param in net.parameters()])  # 输出应小于等于1