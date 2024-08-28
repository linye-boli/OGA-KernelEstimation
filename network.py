import torch
import torch.nn as nn
import torch.nn.functional as F
from siren_pytorch import Sine

class Rational(torch.nn.Module):
    """Rational Activation function.
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self):
        super().__init__()
        # self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.coeffs = torch.Tensor(4, 2)
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([[1.1915, 0.0],
                                    [1.5957, 2.383],
                                    [0.5, 0.0],
                                    [0.0218, 1.0]]).cuda()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output

Activations = {
    'relu' : nn.ReLU,
    'rational' : Rational,
    'tanh' : nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'gelu' : nn.GELU,
    'elu': nn.ELU,
    'sine' : Sine,
}

class ReLUk(nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.k = k 
    
    def forward(self, x):
        return torch.maximum(torch.zeros_like(x), x) ** self.k

class Hat(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.maximum(torch.zeros_like(x), 1-10*(x-0.1).abs())
    
# class Hat(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x, c):
#         return torch.maximum(torch.zeros_like(x), 1-1/c*(x-c).abs())

class Gauss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, c):
        return torch.exp(-10*(x - c)**2)

class Exp(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.exp(x)

class Sine(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(x)


class MMNN(nn.Module):
    def __init__(self, 
                 ranks = [1] + [16]*5 + [1], 
                 widths = [366]*6,
                 device = "cpu", 
                 ResNet = False,
                 fixWb = True):
        super().__init__()
        """
        A class to configure the neural network model.
    
        Attributes:
            ranks (list[int]): A list where the i-th element represents the output dimension of the i-th layer.
                               For the j-th layer, ranks[j-1] is the input dimension and ranks[j] is the output dimension.
            
            widths (list[int]): A list where each element specifies the width of the corresponding layer.
            
            device (str): The device (CPU/GPU) on which the PyTorch code will be executed.
            
            ResNet (bool): Indicates whether to use ResNet architecture, which includes identity connections between layers.
            
            fixWb (bool): If True, the weights and biases are not updated during training.
        """
        
        self.ranks = ranks
        self.widths = widths
        self.ResNet = ResNet
        self.depth = len(widths)
        
        fc_sizes = [ ranks[0] ] 
        for j in range(self.depth):
            fc_sizes += [ widths[j], ranks[j+1] ]

        fcs=[]
        for j in range(len(fc_sizes)-1):
            fc = nn.Linear(fc_sizes[j],
                           fc_sizes[j+1], device=device) 
            # setattr(self, f"fc{j}", fc)
            fcs.append(fc)
        self.fcs = nn.ModuleList(fcs)
        
        if fixWb:
            for j in range(len(fcs)):
                if j % 2 == 0:
                    self.fcs[j].weight.requires_grad = False
                    self.fcs[j].bias.requires_grad = False
 

    def forward(self, x):
        for j in range(self.depth):
            if self.ResNet:
                if 0 < j < self.depth-1:
                    x_id = x + 0
            x = self.fcs[2*j](x)
            x = torch.relu(x)
            x = self.fcs[2*j+1](x) 
            if self.ResNet:
                if 0 < j < self.depth-1:
                    n = min(x.shape[1], x_id.shape[1])
                    x[:,:n] = x[:,:n] + x_id[:,:n]
        return x


# A simple feedforward neural network
class DeepNN(torch.nn.Module):
    def __init__(self, layers, nonlinearity, aug=None):
        super(DeepNN, self).__init__()

        self.n_layers = len(layers) - 1
        self.aug = aug

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != self.n_layers - 1:
                self.layers.append(nonlinearity)

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

# A simple feedforward neural network
class ShallowNN(torch.nn.Module):
    def __init__(self, layers, nonlinearity, aug=None):
        super(ShallowNN, self).__init__()

        nonlinearity = Activations[nonlinearity]
        self.n_layers = len(layers) - 1
        self.aug = aug

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

if __name__ == "__main__":

    import numpy as np
    def f2d(X):
        return 0.5 * (X[:,0] + X[:,1] - np.abs(X[:,0]-X[:,1])) - X[:,0]*X[:,1]

    # generate training data
    X = np.random.rand(5000, 2)
    y = f2d(X)

    x = np.linspace(0,1,101)
    xx, yy = np.meshgrid(x, x)
    Xtest = np.c_[xx.reshape(-1), yy.reshape(-1)]
    ytest = f2d(Xtest)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    Xtest = torch.from_numpy(Xtest).float()
    ytest = torch.from_numpy(ytest).float()
    device = torch.device("cuda:0")

    # init model 
    inpDim = 2
    nLayer = 4 
    nHidden = 50
    act = 'rational'
    dnnfitter = DNNFitter(
        inpDim=inpDim, nLayer=nLayer, nHidden=nHidden, act=act,
        X = X, y=y, Xtest=Xtest, ytest=ytest, device=device)
    
    # fitting 
    adam_niter = 1000
    dnnfitter.optimize_adam(adam_niter, lr=1e-3)

    lbfgs_niter = 1000
    dnnfitter.optimize_lbfgs(lbfgs_niter, lr=1)