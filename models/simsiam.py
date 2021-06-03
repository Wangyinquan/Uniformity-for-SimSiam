import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50

def weight_init(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias,0)
        
class grl_func(torch.autograd.Function):
    def __init__(self):
        super(grl_func, self).__init__()
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GRL(nn.Module):
    def __init__(self, lambda_=0.):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return grl_func.apply(x, self.lambda_)
        


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    elif version == 'revgrad':
#         revgrad = GRL.apply
        return - F.cosine_similarity(p, z, dim=-1).mean()
    else:
        raise Exception



        
class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, model_cfg, hidden_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        layer1,layer2,layer3 = [],[],[]
        if model_cfg.proj_layer == 1:
            dims = [in_dim, out_dim]
            layer1 = nn.Linear(in_dim, out_dim, bias=model_cfg.proj_bias)
        elif  model_cfg.proj_layer == 2:
            dims = [in_dim, hidden_dim, out_dim]
            layer1 = nn.Linear(in_dim, hidden_dim, bias=model_cfg.proj_bias)
            layer2 = nn.Linear(hidden_dim, out_dim, bias=model_cfg.proj_bias)
        elif model_cfg.proj_layer == 3:
            dims = [in_dim, hidden_dim, hidden_dim, out_dim]
            layer1 = nn.Linear(in_dim, hidden_dim, bias=model_cfg.proj_bias)
            layer2 = nn.Linear(hidden_dim, hidden_dim, bias=model_cfg.proj_bias)
            layer3 = nn.Linear(hidden_dim, out_dim, bias=model_cfg.proj_bias)
        
        if model_cfg.proj_bn:
            if layer1!=[]: layer1.add_module('bn',nn.BatchNorm1d(dims[1])) 
            if layer2!=[]: layer2.add_module('bn',nn.BatchNorm1d(dims[2]))
            if layer3!=[]: layer3.add_module('bn',nn.BatchNorm1d(dims[3]))
                
        if model_cfg.proj_relu:
            if layer1!=[]: layer1.add_module('relu',nn.ReLU(inplace=True)) 
            if layer2!=[]: layer2.add_module('relu',nn.ReLU(inplace=True))
            
        self.net = nn.Sequential(layer1)
        if model_cfg.proj_layer >= 2: 
            self.net.add_module('layer2',layer2)
        if model_cfg.proj_layer >= 3: 
            self.net.add_module('layer3',layer3)
            
    def forward(self, x):
        x = self.net(x)
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, outdim, model_cfg): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        in_dim=outdim
        hidden_dim=512
        out_dim=outdim
        
        layer1,layer2,layer3 = [],[],[]
        if model_cfg.pred_layer == 1:
            dims = [in_dim, out_dim]
            layer1 = nn.Linear(in_dim, out_dim, bias=model_cfg.pred_bias)
        elif  model_cfg.pred_layer == 2:
            dims = [in_dim, hidden_dim, out_dim]
            layer1 = nn.Linear(in_dim, hidden_dim, bias=model_cfg.pred_bias)
            layer2 = nn.Linear(hidden_dim, out_dim, bias=model_cfg.pred_bias)
        elif model_cfg.pred_layer == 3:
            dims = [in_dim, hidden_dim, hidden_dim, out_dim]
            layer1 = nn.Linear(in_dim, hidden_dim, bias=model_cfg.pred_bias)
            layer2 = nn.Linear(hidden_dim, hidden_dim, bias=model_cfg.pred_bias)
            layer3 = nn.Linear(hidden_dim, out_dim, bias=model_cfg.pred_bias)
        
        if model_cfg.pred_bn:
            if layer1!=[] and model_cfg.pred_layer != 1: layer1.add_module('bn',nn.BatchNorm1d(dims[1])) 
            if layer2!=[] and model_cfg.pred_layer != 2: layer2.add_module('bn',nn.BatchNorm1d(dims[2]))
#             if layer3!=[]: layer3.add_module('bn',nn.BatchNorm1d(dims[3]))
                
        if model_cfg.pred_relu:
            if layer1!=[] and model_cfg.pred_layer != 1: layer1.add_module('relu',nn.ReLU(inplace=True)) 
            if layer2!=[] and model_cfg.pred_layer != 2: layer2.add_module('relu',nn.ReLU(inplace=True))
            
        self.net = nn.Sequential(layer1)
        if model_cfg.pred_layer >= 2: 
            self.net.add_module('layer2',layer2)
        if model_cfg.pred_layer >= 3: 
            self.net.add_module('layer3',layer3)

    def forward(self, x):
        x = self.net(x)
        return x 

class SimSiam(nn.Module):
    def __init__(self, model_cfg, backbone=resnet50(),outdim=2048):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim, outdim, model_cfg)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP(outdim, model_cfg)
        self.cfg = model_cfg
        self.version = 'simplified'
        if hasattr(self.cfg,'version') and self.cfg.version == 'revgrad':
            self.GRL = GRL(1)
            self.version = self.cfg.version
    def forward(self, x1, x2):

        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        if hasattr(self.cfg,'z_div'):
            z1 = z1/float(self.cfg.z_div)
            z2 = z2/float(self.cfg.z_div)
        p1, p2 = h(z1), h(z2)
        if hasattr(self.cfg,'version') and self.cfg.version == 'revgrad':
            z1 = self.GRL(z1)
            z2 = self.GRL(z2)
        L = D(p1, z2, self.version) / 2 + D(p2, z1, self.version) / 2
        return {'loss': L}



if __name__ == "__main__":
    model = SimSiam()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)

# Output:
# tensor(-0.0010)
# 0.005159854888916016
# tensor(-0.0010)
# 0.0014872550964355469












