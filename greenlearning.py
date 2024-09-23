import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import relative_err
from utils import init_records
from utils import UnitGaussianNormalizer

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

class GL(nn.Module):
    def __init__(self, inpDim):
        super(GL, self).__init__()
        self.fc1 = nn.Linear(inpDim,50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,50)
        self.fc4 = nn.Linear(50,50)
        self.fc5 = nn.Linear(50,1)
        self.R1 = Rational()
        self.R2 = Rational()
        self.R3 = Rational()
        self.R4 = Rational()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.R1(x)
        x = self.fc2(x)
        x = self.R2(x)
        x = self.fc3(x)
        x = self.R3(x)
        x = self.fc4(x)
        x = self.R4(x)
        x = self.fc5(x)
        return x

class GLFitter:
    def __init__(
            self,
            X, fs, us, fTest, uTest, Gref=None, 
            device=None):
        
        '''
        X  : nPts x (nx + ny)
        fs : nPts x nSample 
        us : nPts x nSample 
        Gref : nPts x nPts
        '''

        if X.shape[1] == 4:
            area = np.pi 
        else:
            area = 1

        # dataset 
        self.nyPts = fs.shape[0]
        self.nxPts = us.shape[0]
        self.h = area/self.nyPts
        self.inpDim = X.shape[1]
        self.X = X
        self.fs = fs 
        self.us = us 
        self.fTest = fTest 
        self.uTest = uTest
        self.Gref = Gref

        self.Gk = np.zeros((self.nxPts, self.nyPts))

        # neural network
        self.model = GL(self.inpDim)
        print()
        print("model structure")
        print(self.model)
        print()

        # log 
        self.Glog = []
        self.utest_log = []
        self.utrain_log = []
        
        # put to device 
        self.device = device 
        self.to_device()        
    
    def to_device(self):
        self.X = torch.from_numpy(self.X).float()
        self.fs = torch.from_numpy(self.fs).float()
        self.us = torch.from_numpy(self.us).float()
        self.fTest = torch.from_numpy(self.fTest).float()
        self.uTest = torch.from_numpy(self.uTest).float()

        self.X = self.X.to(self.device)
        self.fs = self.fs.to(self.device)
        self.us = self.us.to(self.device)
        self.fTest = self.fTest.to(self.device)
        self.uTest = self.uTest.to(self.device)
        self.model = self.model.to(self.device)

    def optimize_adam(self, niter, lr=1e-3, dispstep=10):
        self.opt_adam = torch.optim.Adam(self.model.parameters(), lr)
        self.sch = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.opt_adam, milestones=[1000, 3000], gamma=1)

        print()
        print("Adam optimization start")
        print()

        for k in range(niter):
            self.model.train()
            self.opt_adam.zero_grad()
            Gk = self.model(self.X)
            self.Gk = Gk.reshape(self.nxPts, self.nyPts)
            utrain_Pred = self.h * self.Gk @ self.fs
            loss = relative_err(utrain_Pred.T, self.us.T)
            loss.backward()
            self.opt_adam.step()
            self.sch.step()

            if self.Gref is not None:
                Grl2 = relative_err(self.Gk.detach().cpu().reshape(-1,1).numpy(), self.Gref.reshape(-1,1))
                self.Glog.append(Grl2)
            else:
                Grl2 = 1.0

            self.model.eval()
            with torch.no_grad():
                Gk = self.model(self.X)
            utest_Pred = self.h * self.Gk @ self.fTest
            utrain_rl2 = relative_err(utrain_Pred.detach().T.cpu().numpy(), self.us.T.cpu().numpy())
            utest_rl2 = relative_err(utest_Pred.detach().T.cpu().numpy(), self.uTest.T.cpu().numpy())

            self.utest_log.append(utest_rl2)
            self.utrain_log.append(utrain_rl2)
            
            if k % dispstep == 0:
                print('{:}th train url2 : {:.4e} - test url2 : {:.4e} - Grl2 : {:.4e} - learning rate : {:.4e}'.format(k, utrain_rl2, utest_rl2, Grl2, self.sch.get_last_lr()[0]))

        print('{:}th url2 : {:.4e} - Grl2 : {:.4e}'.format(k, utest_rl2, Grl2))        

        self.Gk = {"X": self.X.cpu().numpy(), "Gpred" : self.Gk.detach().cpu().numpy(), "Gref":self.Gref}
        self.utest_Pred = utest_Pred.detach().cpu().numpy()
        # self.model_weights = {"WB":self.WB.detach().cpu().numpy(), "Alpha":self.Alpha.detach().cpu().numpy()}
        self.log = {"G_rl2" : np.array(self.Glog), "utest_rl2" : np.array(self.utest_log), "utrain_rl2" : np.array(self.utrain_log)}
        # print(self.Alpha)
    
    def optimize_adam_batch(self, niter, lr=1e-3, dispstep=100):
        self.opt_adam = torch.optim.Adam(self.model.parameters(), lr)
   
        print()
        print("Adam optimization start")
        print()

        for k in range(niter):
            self.opt_adam.zero_grad()
            Gk = self.model(self.X)
            self.Gk = Gk.reshape(self.nxPts, self.nyPts)
            utrain_Pred = self.h * self.Gk @ self.fs
            loss = relative_err(utrain_Pred.T, self.us.T)
            loss.backward()
            self.opt_adam.step()

            if self.Gref is not None:
                Grl2 = relative_err(self.Gk.detach().cpu().reshape(-1,1).numpy(), self.Gref.reshape(-1,1))
                self.Glog.append(Grl2)
            else:
                Grl2 = 1.0

            utest_Pred = self.h * self.Gk @ self.fTest
            utrain_rl2 = relative_err(utrain_Pred.detach().T.cpu().numpy(), self.us.T.cpu().numpy())
            utest_rl2 = relative_err(utest_Pred.detach().T.cpu().numpy(), self.uTest.T.cpu().numpy())

            self.utest_log.append(utest_rl2)
            self.utrain_log.append(utrain_rl2)
            
            if k % dispstep == 0:
                print('{:}th train url2 : {:.4e} - test url2 : {:.4e} - Grl2 : {:.4e}'.format(k, utrain_rl2, utest_rl2, Grl2))

        print('{:}th url2 : {:.4e} - Grl2 : {:.4e}'.format(k, utest_rl2, Grl2))        

        self.Gk = {"X": self.X.cpu().numpy(), "Gpred" : self.Gk.detach().cpu().numpy(), "Gref":self.Gref}
        self.utest_Pred = utest_Pred.detach().cpu().numpy()
        # self.model_weights = {"WB":self.WB.detach().cpu().numpy(), "Alpha":self.Alpha.detach().cpu().numpy()}
        self.log = {"G_rl2" : np.array(self.Glog), "utest_rl2" : np.array(self.utest_log), "utrain_rl2" : np.array(self.utrain_log)}
        # print(self.Alpha)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGA for kernel estimation')
    parser.add_argument('--task', type=str, default='poisson1D',
                        help='dataset name. (poisson1D, helmholtz1D, airy1D, poisson2D)')
    parser.add_argument('--nIter', type=int, default=2000, 
                        help='maximum number of neurons')
    parser.add_argument('--nTrain', type=int, default=200, 
                        help='number of training samples')
    parser.add_argument('--nTest', type=int, default=200, 
                        help='number of test samples')
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}')
    # device = torch.device('cpu')
        
    if args.task == 'poisson1D':
        from utils import load_poisson1d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_poisson1d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'helmholtz1D':
        from utils import load_helmholtz1d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmholtz1d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'poisson2D':
        from utils import load_poisson2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_poisson2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'helmholtz2D':
        from utils import load_helmholtz2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmholtz2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'poisson2Dhdomain':
        from utils import load_poisson2dhdomain_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_poisson2dhdomain_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'helmholtz2D':
        from utils import load_helmholtz2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmholtz2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'helmholtz2Dhdomain':
        from utils import load_helmholtz2dhdomain_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmholtz2dhdomain_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'log3D':
        from utils import load_log3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_log3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'logsin3D':
        from utils import load_logsin3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_logsin3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'cos3D':
        from utils import load_cos3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_cos3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'logcos3D':
        from utils import load_logcos3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_logcos3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)

    # import pdb 
    # pdb.set_trace()

    model = GLFitter(X=X, 
        fs=fTrain, us=uTrain, 
        fTest=fTest, uTest=uTest, 
        Gref=Gref, device=device)

    # model train
    if "3D" in args.task:
        model.optimize_adam_batch(args.nIter)
    elif "1D" in args.task:
        model.optimize_adam(args.nIter)
    elif "2D" in args.task:
        model.optimize_adam(args.nIter)

    # # save outputs
    log_outpath, upred_outpath, model_outpath, Gpred_outpath = init_records('./results', args.task, 'gl-{:}-{:}'.format(args.nIter, args.nTrain))

    np.save(log_outpath, model.log)
    np.save(upred_outpath, model.utest_Pred)
    np.save(Gpred_outpath, model.Gk)
    # np.save(model_outpath, model.model_weights)
