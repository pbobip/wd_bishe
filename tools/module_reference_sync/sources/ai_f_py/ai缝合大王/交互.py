import torch
from torch import nn
from einops.einops import rearrange
from 模块制作.串联 import LKA, AKConv


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

#CNN-branch
class AKLKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lka = LKA(dim)
        self.ak = AKConv(inc=dim, outc=dim, num_param=3)
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        res = x
        x = self.ak(x)
        x = self.lka(x)
        x = self.conv(x)
        return x + res


#tansformer-branch
class MobileViTv2Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model, 1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.d_model = d_model

    def forward(self, input):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        '''
        i = self.fc_i(input)  # (bs,nq,1)
        weight_i = torch.softmax(i, dim=1)  # bs,nq,1
        context_score = weight_i * self.fc_k(input)  # bs,nq,d_model
        context_vector = torch.sum(context_score, dim=1, keepdim=True)  # bs,1,d_model
        v = self.fc_v(input) * context_vector  # bs,nq,d_model
        out = self.fc_o(v)  # bs,nq,d_model

        return out

#融合
class fu(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim,dim,1)
        self.conv2 = nn.Conv2d(dim,dim,1)
        self.act = nn.ReLU()
    def forward(self, x):
        res = x
        x=self.conv1(x)
        x =self.act(x)
        x=self.conv2(x)
        return x + res

# 输入 N C H W,  输出 N C H W

class ctf(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv=AKLKA(dim)
        self.att=MobileViTv2Attention(dim)
        self.fu=fu(dim*2)
        self.split_indexes = (dim, dim)
    def forward(self, x1,x2):
        b,c,h,w=x2.size()

        x1 =self.conv(x1)

        x2=to_3d(x2)
        x2 =self.att(x2)
        x2 =to_4d(x2,h,w)

        x=torch.cat((x1,x2),dim=1)
        x=self.fu(self.fu(x))+x
        x1, x2 = torch.split(x, self.split_indexes, dim=1)
        return x1,x2
if __name__ == '__main__':
    block = ctf(64)
    input = torch.rand(3, 64, 56, 56)
    output1,output2 = block(input,input)
    print(input.size(), output1.size(),output2.size())
