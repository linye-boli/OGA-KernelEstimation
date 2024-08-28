import os 
import scipy 
import numpy as np
import argparse
import torch 

from utils import relative_err
from network import MMNN, ReLUk, Rational, Hat

class MMNNFitter:
    def __init__(
            self, 
            nLayer,
            nRank,
            nWidth,
            ResNet,
            X, fs, us, fTest, uTest, 
            Gref=None, device=None):
        
        if X.shape[1] == 4:
            # Rx = (X[:,[0]]**2 + X[:,[1]]**2) ** 0.5
            # Ry = (X[:,[2]]**2 + X[:,[3]]**2) ** 0.5
            # Dxy = ((X[:,[0]] - X[:,[2]])**2 + (X[:,[1]]-X[:,[3]])**2) ** 0.5
            # X = np.c_[Rx, Ry, Dxy]
            area = np.pi 
        else:
            area = 1


        # dataset 
        self.X = X 
        self.fs = fs 
        self.us = us 
        self.fTest = fTest 
        self.uTest = uTest
        self.Gref = Gref
        self.nyPts = fs.shape[0]
        self.nxPts = us.shape[0]
        self.h = area/self.nyPts

        # dnn config
        self.inpDim = X.shape[1]
        self.nLayer = nLayer 
        self.widths = [nWidth] * (nLayer + 1) 
        self.ranks = [self.inpDim] + [nRank] * nLayer + [1]
        self.ResNet = ResNet
        self.device = device 
        self.model = MMNN(
            ranks=self.ranks, 
            widths=self.widths, 
            device=device,
            ResNet=ResNet)
        print()
        print("model structure")
        print(self.model)
        print()
        
        # put to device 
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

    def optimize_adam(self, niter, lr=1e-3, dispstep=1):
        self.opt_adam = torch.optim.Adam(self.model.parameters(), lr)
        self.sch_adam = torch.optim.lr_scheduler.StepLR(
            optimizer=self.opt_adam, step_size=2000, gamma=0.1)

        print()
        print("Adam optimization start")
        print()

        for n in range(niter):
            self.model.train()
            self.opt_adam.zero_grad()
            Gpred = self.model(self.X).reshape(self.nxPts, self.nyPts)
            upred = self.h * Gpred @ self.fs

            loss = ((self.us-upred)**2).mean()
            loss.backward()
            self.opt_adam.step()
            self.sch_adam.step()

            if n % dispstep == 0:
                with torch.no_grad():
                    train_rl2 = relative_err(upred.T.detach().cpu().numpy(), self.us.T.cpu().numpy())
                    Gpred = self.model(self.X).reshape(self.nxPts, self.nyPts)
                    upred = self.h * Gpred @ self.fTest
                    test_rl2 = relative_err(upred.T.detach().cpu().numpy(), self.uTest.T.cpu().numpy())

                print("{:}th - train mse : {:.4e} - train rl2 : {:.4e} - val rl2 : {:.4e}".format(
                    n, loss.item(), train_rl2, test_rl2))

        self.model.eval()
        with torch.no_grad():
            Gpred = self.model(self.X).reshape(self.nxPts, self.nyPts)
            upred = self.h * Gpred @ self.fTest
            test_rl2 = relative_err(upred.T.detach().cpu().numpy(), self.uTest.T.cpu().numpy())

        print("{:}th - train mse : {:.4e} - val rl2 : {:.4e}".format(n, loss.item(), test_rl2))
        print("Adam optimization finished")
        print()
    
    def optimize_lbfgs(self, niter, lr=1e-3, dispstep=10):
        self.opt_lbfgs = torch.optim.LBFGS(self.model.parameters(), lr)
        
        print()
        print("L-BFGS optimization start")
        print()
        
        for n in range(niter):
            def closure():
                self.opt_lbfgs.zero_grad()
                Gpred = self.model(self.X).reshape(self.nxPts, self.nyPts)
                upred = self.h * Gpred @ self.fs
                loss = ((self.us-upred)**2).mean()
                loss.backward()
                return loss 

            loss_prev = closure()
            self.opt_lbfgs.step(closure)
            loss_cur = closure()
            
            if n % dispstep == 0:
                self.model.eval()
                with torch.no_grad():
                    Gpred = self.model(self.X).reshape(self.nxPts, self.nyPts)
                    upred = self.h * Gpred @ self.fTest
                    test_rl2 = relative_err(upred.detach().cpu().numpy(), self.uTest.cpu().numpy())
                print("{:}th - train mse : {:.4e} - val rl2 : {:.4e}".format(n, loss_cur.item(), test_rl2))

            if (loss_prev - loss_cur).abs() < 1e-16:
                break 

        self.model.eval()
        with torch.no_grad():
            Gpred = self.model(self.X).reshape(self.nxPts, self.nyPts)
            upred = self.h * Gpred @ self.fTest
            test_rl2 = relative_err(upred.detach().cpu().numpy(), self.uTest.cpu().numpy())
        print("{:}th - train mse : {:.4e} - val rl2 : {:.4e}".format(n, loss_cur.item(), test_rl2))
        print("L-BFGS optimization finished")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dnn function fitting')
    parser.add_argument('--task', type=str, default='poisson1d',
                        help='dataset name. (sin, poisson1d, helmholtz2d, poisson2d)')
    parser.add_argument('--act', type=str, default='rational',
                        help='activation name. (sin, relu, rational, tanh, sigmoid)')
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    parser.add_argument('--nTrain', type=int, default=200,
                        help='number of train samples.')
    parser.add_argument('--nTest', type=int, default=200,
                        help='number of test samples.')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}')

    if args.task == 'poisson1D':
        from utils import load_poisson1d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_poisson1d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    if args.task == 'gabor1D':
        from utils import load_gabor1d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_gabor1d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'helmholtz1D':
        from utils import load_helmholtz1d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmholtz1d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'ad1D':
        from utils import load_ad1d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_ad1d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'poisson2D':
        from utils import load_poisson2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_poisson2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=20)
    elif args.task == 'helmholtz2D':
        from utils import load_helmholtz2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmholtz2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=20)
    elif args.task == 'cosine2D':
        from utils import load_cosine2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_cosine2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=20)
    elif args.task == 'log2D':
        from utils import load_log2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_log2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=20)
    elif args.task == 'helmreal2D':
        from utils import load_helmreal2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmreal2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=20)
    elif args.task == 'helmimg2D':
        from utils import load_helmimg2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmimg2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=20)
    elif args.task == 'gauss2D':
        from utils import load_gauss2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_gauss2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=20)
    elif args.task == 'sine2D':
        from utils import load_sine2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_sine2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=20)


    if args.act == 'relu':
        act = ReLUk(k=1)
    elif args.act == 'relu4':
        act = ReLUk(k=4)
    elif args.act == 'rational':
        act = Rational()
    elif args.act == 'hat':
        act = Hat()

    # network prepare
    model = MMNNFitter(nLayer=8, nRank=36, nWidth=666, ResNet=True,
                       X = X, fs=fTrain, us=uTrain, fTest=fTest, uTest=uTest, 
                       Gref=Gref, device=device)

    # parameters for poisson2D fitting 
    # fitting 
    adam_niter = 5000
    model.optimize_adam(adam_niter, lr=1e-3)

    # lbfgs_niter = 10000
    # model.optimize_lbfgs(lbfgs_niter, lr=1e-1)