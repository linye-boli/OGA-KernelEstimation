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
            X, y, Xtest, ytest,
            device):

        # dataset 
        self.X = X 
        self.y = y
        self.Xtest = Xtest 
        self.ytest = ytest 

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
        self.y = torch.from_numpy(self.y).float()
        self.Xtest = torch.from_numpy(self.Xtest).float()
        self.ytest = torch.from_numpy(self.ytest).float()

        self.X = self.X.to(self.device)
        self.y = self.y.to(self.device)
        self.Xtest = self.Xtest.to(self.device)
        self.ytest = self.ytest.to(self.device)

        self.model = self.model.to(self.device)

    def optimize_adam(self, niter, lr=1e-3, dispstep=100):
        self.opt_adam = torch.optim.Adam(self.model.parameters(), lr)

        print()
        print("Adam optimization start")
        print()

        for n in range(niter):
            self.opt_adam.zero_grad()
            ypred = self.model(self.X)
            loss = ((self.y-ypred)**2).mean()
            loss.backward()
            self.opt_adam.step()

            if n % dispstep == 0:
                with torch.no_grad():
                    ypred = self.model(self.X)
                    train_rl2 = relative_err(ypred.detach().cpu().numpy(), self.y.cpu().numpy())

                    ypred = self.model(self.Xtest)
                    test_rl2 = relative_err(ypred.detach().cpu().numpy(), self.ytest.cpu().numpy())

                print("{:}th - train mse : {:.4e} - train rl2 : {:.4e} - val rl2 : {:.4e}".format(
                    n, loss.item(), train_rl2, test_rl2))

        self.model.eval()
        with torch.no_grad():
            ypred = self.model(self.Xtest)
            rl2 = relative_err(ypred.detach().cpu().numpy(), self.ytest.cpu().numpy())
        self.ypred_adam = ypred
        print("{:}th - train mse : {:.4e} - val rl2 : {:.4e}".format(n, loss.item(), rl2))
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
                ypred = self.model(self.X)
                loss = ((ypred - self.y)**2).mean()
                loss.backward()
                return loss 

            loss_prev = closure()
            self.opt_lbfgs.step(closure)
            loss_cur = closure()
            
            if n % dispstep == 0:
                self.model.eval()
                with torch.no_grad():
                    ypred = self.model(self.Xtest)
                    rl2 = relative_err(ypred.detach().cpu().numpy(), self.ytest.cpu().numpy())
                print("{:}th - train mse : {:.4e} - val rl2 : {:.4e}".format(n, loss_cur.item(), rl2))

            if (loss_prev - loss_cur).abs() < 1e-16:
                break 

        self.model.eval()
        with torch.no_grad():
            ypred = self.model(self.Xtest)
            rl2 = relative_err(ypred.detach().cpu().numpy(), self.ytest.cpu().numpy())
        self.ypred_lbfgs = ypred
        print("{:}th - train mse : {:.4e} - val rl2 : {:.4e}".format(n, loss_cur.item(), rl2))
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
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}')

    if args.task == 'poisson1d':
        from utils import load_poisson1d_fitting_dataset
        X, y, Xtest, ytest = load_poisson1d_fitting_dataset()
    elif args.task == 'poisson1d':
        from utils import load_poisson1d_fitting_dataset
        X, y, Xtest, ytest = load_poisson1d_fitting_dataset()
    elif args.task == 'poisson2d':
        from utils import load_poisson2d_fitting_dataset
        X, y, Xtest, ytest = load_poisson2d_fitting_dataset()
    elif args.task == 'rbf2d':
        from utils import load_rbf2d_fitting_dataset
        X, y, Xtest, ytest = load_rbf2d_fitting_dataset()
    elif args.task.startswith('rbf1d'):
        L = int(args.task.split('-')[1])
        from utils import load_rbf1d_fitting_dataset
        X, y, Xtest, ytest = load_rbf1d_fitting_dataset(L=L)  
    elif args.task.startswith('runge1d'):
        L = int(args.task.split('-')[1])
        from utils import load_runge1d_fitting_dataset
        X, y, Xtest, ytest = load_runge1d_fitting_dataset(L=L) 

    if args.act == 'relu':
        act = ReLUk(k=1)
    elif args.act == 'relu4':
        act = ReLUk(k=4)
    elif args.act == 'rational':
        act = Rational()
    elif args.act == 'hat':
        act = Hat()
    

    # network prepare
    model = MMNNFitter(nLayer=8, nRank=36, nWidth=666, ResNet=False,
        X = X, y=y, Xtest=Xtest, ytest=ytest, device=device)

    # parameters for poisson2D fitting 
    # fitting 
    adam_niter = 1000
    model.optimize_adam(adam_niter, lr=1e-5)

    lbfgs_niter = 10000
    model.optimize_lbfgs(lbfgs_niter, lr=1)