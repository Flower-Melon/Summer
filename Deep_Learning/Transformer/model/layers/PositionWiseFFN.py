from torch import nn

class PositionWiseFFN(nn.Module):
    """"基于位置的前馈网络层"""
    def __init__(self,num_inputs,num_hiddens,num_outputs):
        super(PositionWiseFFN,self).__init__()
        self.dense1 = nn.Linear(num_inputs,num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(num_hiddens,num_outputs)
    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))
    
if __name__ == "__main__":
    """测试PositionWiseFFN"""
    import torch as T
    ffn = PositionWiseFFN(4, 4, 8)
    ffn.eval()
    ffn(T.ones((2, 3, 4)))[0]