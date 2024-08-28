import argparse
import os 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import scipy 
from scipy.optimize import minimize

from utils import PoissonKernelDisk, relative_err
from utils import init_records, save_pytorch_model


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
        self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([[1.1915, 0.0],
                                    [1.5957, 2.383],
                                    [0.5, 0.0],
                                    [0.0218, 1.0]])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output

def Rational_np(x, coeffs):
    coeffs[0,1] = 0
    x = np.power(np.expand_dims(x, axis=-1), [3,2,1,0])
    PQ = x @ coeffs
    y = PQ[...,0]/PQ[...,1]
    return y 

class GL2D:
    def __init__(self, nLayer, nHidden, x, y, fs, us, fTest, uTest, area, Gref=None):
        self.nLayer = nLayer 
        self.nHidden = nHidden 
        self.inpDim = x.shape[1] + y.shape[1]
        self.layers = [nHidden] * nLayer 
        self.nxPts = x.shape[0]
        self.nyPts = y.shape[0]


        # mesh 
        idx = np.arange(self.nxPts)
        idy = np.arange(self.nyPts)
        idxx, idyy = np.meshgrid(idx, idy)
        xs = x[idxx.reshape(-1)]
        ys = y[idyy.reshape(-1)]
        self.x1s = xs[:,0]
        self.x2s = xs[:,1]        
        self.y1s = ys[:,0]
        self.y2s = ys[:,1]
        self.h = area / self.nyPts

        # dataset             
        Gref = PoissonKernelDisk(xs, ys).reshape(self.nxPts, self.nyPts)
        self.Gref = Gref
        self.X = np.c_[xs, ys]
        self.fs = fs 
        self.fTest = fTest 
        self.us = us 
        self.uTest = uTest
        
        self.us = self.h * Gref @ fs        
        self.uTest = self.h * Gref @ fTest

        # model 
        self.deep_nn = self.init_dnn()
    
    def init_dnn(self):
        layer_lst = []
        nLayer = self.nLayer
        for i in range(nLayer):
            if i == 0:
                layer_lst.append(nn.Linear(self.inpDim, self.layers[i]))
                # layer_lst.append(nn.ReLU())
                layer_lst.append(Rational())
            if (i > 0) & (i < nLayer-1):                
                layer_lst.append(nn.Linear(self.layers[i], self.layers[i+1]))
                # layer_lst.append(nn.ReLU())
                layer_lst.append(Rational())
            if i == nLayer-1:
                layer_lst.append(nn.Linear(self.layers[i], 1))
        return nn.Sequential(*layer_lst)
    
    def to_tensor(self, cuda=False):
        if cuda:
            self.X = torch.from_numpy(self.X).float().cuda()
            self.fs = torch.from_numpy(self.fs).float().cuda()
            self.us = torch.from_numpy(self.us).float().cuda()
            self.fTest = torch.from_numpy(self.fTest).float().cuda()
            self.uTest = torch.from_numpy(self.uTest).float().cuda()
        else:
            self.X = torch.from_numpy(self.X).float()
            self.fs = torch.from_numpy(self.fs).float()
            self.us = torch.from_numpy(self.us).float()
            self.fTest = torch.from_numpy(self.fTest).float()
            self.uTest = torch.from_numpy(self.uTest).float()
    
    def to_numpy(self):
        self.X = self.X.cpu().numpy()
        self.fs = self.fs.cpu().numpy()
        self.us = self.us.cpu().numpy()
        self.fTest = self.fTest.cpu().numpy()
        self.uTest = self.uTest.cpu().numpy()

        WBs = []
        coeffs = []
        for i in range(len(self.deep_nn)):
            if i % 2 ==0:
                # parameters of linear layers
                weight = self.deep_nn[i].weight.detach().cpu().numpy()
                bias = self.deep_nn[i].bias
                if bias is None:
                    WBs.append([weight])
                else:
                    bias = bias.detach().cpu().numpy()
                    WBs.append([weight, bias])
            else:
                # parameters of rational activations
                coeff = self.deep_nn[i].coeffs.detach().cpu().numpy()
                coeffs.append(coeff)

        self.WBs = WBs
        self.Cs = coeffs
    
    def np_forward(self, X):
        for i in range(len(self.deep_nn)):
            if i % 2 == 0:
                if len(self.WBs[i//2]) == 2:
                    Wi, Bi = self.WBs[i//2]
                else:
                    Wi, Bi = self.WBs[i//2][0], 0
                
                if i == 0:
                    V = X @ Wi.T + Bi 
                else:
                    V = V @ Wi.T + Bi
            else:
                V = Rational_np(V, self.Cs[i//2])

        return V
    
    def params_concat(self):
        params_wb = []
        split_wb_idx = []

        wb_idx = 0
        for i in range(self.nLayer):
            if i < self.nLayer -1:
                Wi, Bi = self.WBs[i]
                Bi = Bi.reshape(-1,1)
                params_wb.append(Wi)
                params_wb.append(Bi)

                wb_idx += Wi.shape[1]
                split_wb_idx.append(wb_idx)
                wb_idx += Bi.shape[1]
                split_wb_idx.append(wb_idx)
            else:
                Wi = self.WBs[i][0].T
                params_wb.append(Wi)
            
        params_wb = np.concatenate(params_wb, axis=1).reshape(-1,)
        self.split_wb_idx = split_wb_idx
        self.split_idx = params_wb.shape[0]

        params_c = []
        split_c_idx = []
        c_idx = 0
        for i in range(self.nLayer-1):
            Ci = self.Cs[i]
            params_c.append(Ci)
            c_idx += Ci.shape[1]
            if i < self.nLayer-2:
                split_c_idx.append(c_idx)

        params_c = np.concatenate(params_c, axis=1).reshape(-1,)
        self.split_c_idx = split_c_idx

        params = np.r_[params_wb, params_c]       
        return params

    def params_split(self, params):
        params_wb = params[:self.split_idx]
        params_c = params[self.split_idx:]

        # split wb parameters
        params_wb = np.split(
            params_wb.reshape(self.nHidden,-1), self.split_wb_idx, axis=1)
        for i in range(self.nLayer):
            if i < self.nLayer -1:
                Wi = params_wb[2*i]
                Bi = params_wb[2*i + 1].reshape(-1)
                self.WBs[i] = [Wi, Bi]
            else:
                Wi = params_wb[2*i].T
                self.WBs[i] = [Wi]

        # split coeff parameters
        params_c = np.split(
            params_c.reshape(4,-1), self.split_c_idx, axis=1)
        
        for i in range(self.nLayer-1):
            self.Cs[i] = params_c[i]

    def optimize_adam(self, niter=2500, cuda=False, dispstep=100, lr=1e-3):
        # load data and weights to GPU
        self.to_tensor(cuda)
        if cuda:
            self.deep_nn.cuda()
        print('put data and model to device')

        print('model structure')
        print(self.deep_nn)

        # init optimizer 
        optimizer = optim.Adam(self.deep_nn.parameters(), lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
        print()
        print('start adam optimization')
        print()

        # optimization loop        
        for n in range(niter):
            self.deep_nn.train()
            optimizer.zero_grad()
            Gpred = self.deep_nn(self.X).reshape(self.nxPts, self.nyPts)
            upred = self.h * Gpred @ self.fs
            cost = ((self.us - upred)**2).mean()
            cost.backward()
            optimizer.step()
            # scheduler.step()

            if n % dispstep == 0:
                self.deep_nn.eval()
                with torch.no_grad():
                    upred = self.h * Gpred @ self.fTest
                Grl2 = relative_err(Gpred.detach().cpu().numpy().reshape(1,-1), self.Gref.reshape(1,-1))
                url2 = relative_err(upred.detach().cpu().numpy().T, self.uTest.T.cpu().numpy())
                # lr = scheduler.get_last_lr()[0]
                print("{:}th : Grl2-{:.4e} url2-{:.4e}".format(n, Grl2, url2))
        
        # upred = self.h * Gpred @ self.fTest
        # upred_torch = upred.detach().cpu().numpy()

        self.to_numpy()
        # Gpred = self.np_forward(self.X).reshape(self.nxPts, self.nyPts)
        # upred_np = self.h * Gpred @ self.fTest

        # print(RelativeErr(upred_torch.T, self.uTest.T))
        # print(RelativeErr(upred_np.T, self.uTest.T))

        # params = self.params_concat()
        # self.params_split(params)
        # Gpred = self.np_forward(self.X).reshape(self.nxPts, self.nyPts)
        # upred_np = self.h * Gpred @ self.fTest
        # print(RelativeErr(upred_np.T, self.uTest.T))
        
    def optimize_lbfgs(self):
        Params = self.params_concat().reshape(-1,)
        print()
        print('start l-bfgs optimization')
        print()

        optLog = minimize(self.loss, Params,
                             method='L-BFGS-B',
                             options={'maxiter': 20, #5* 10**4,
                                      'maxfun': 10**5,
                                      'iprint': 10,
                                      'maxcor': 50,
                                      'maxls': 50,
                                      'ftol': 1e-20,
                                      'gtol': 1.0 * np.finfo(float).eps})

        self.params_split(optLog.x)
        return optLog
    
    def loss(self, x):
        self.params_split(x)        
        Gpred = self.np_forward(self.X).reshape(self.nxPts, self.nyPts)
        upred = self.h * Gpred @ self.fs
        J = ((self.us - upred)**2).mean()
        return J


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepGreen for 2D kernel estimation')
    parser.add_argument('--task', type=str, default='helmholtz2D',
                        help='dataset name. (poisson2D, helmholtz2D)')
    parser.add_argument('--nTrain', type=int, default=10, 
                        help='number of training samples')
    parser.add_argument('--nTest', type=int, default=200, 
                        help='number of test samples')
    parser.add_argument('--res', type=int, default=20, 
                        help='mesh resolution')
    args = parser.parse_args()

    # load dataset
    data_root = './data'
    mesh_path = os.path.join(data_root, 'mesh2D_{:}.mat'.format(args.res))
    meshy_path = os.path.join(data_root, 'mesh2Dy_{:}.mat'.format(args.res))
    fs_path = os.path.join(data_root, 'dat2Dy_{:}.mat'.format(args.res))
    us_path = os.path.join(data_root, f'{args.task}_{args.res}.mat')

    xs = scipy.io.loadmat(mesh_path)['X']
    ys = scipy.io.loadmat(meshy_path)['Y']
    fs = scipy.io.loadmat(fs_path)['Fy']
    us = scipy.io.loadmat(us_path)['U']

    nTrain = args.nTrain
    nTest = args.nTest

    fTrain = fs[:,:nTrain]
    fTest = fs[:,-nTest:]
    uTrain = us[:,:nTrain]
    uTest = us[:,-nTest:]


    # network prepare
    nLayer = 5 
    nHidden = 50
    nSample = nTrain
    area = np.pi
    model = GL2D(nLayer, nHidden, xs, ys, fTrain, uTrain, fTest, uTest, area)
    # model train
    model.optimize_adam(niter=20000, lr=1e-3, dispstep=100, cuda=True)
    # save_pytorch_model('./results', args.task, 'pre-gl-rational', model.deep_nn)
    model.optimize_lbfgs()

    # # np.save(log_outpath, model.log)
    # torch.save(model_outpath, model.deep_nn)