import argparse
import numpy as np
import torch 
torch.set_default_dtype(torch.float64)

from utils import relative_err
from utils import init_records
from network import ReLUk, Gauss, Exp, Rational

class ShallowOGAFitter:
    def __init__(
            self, 
            nNeuron,
            X, y, Xtest, ytest,
            device):
        
        # dataset 
        self.nPts = X.shape[0]
        self.inpDim = X.shape[1]
        self.X = X
        self.y = y
        self.yk = np.zeros_like(y)
        self.Xtest = Xtest
        self.ytest = ytest
        self.h = 1/self.nPts
        self.ylog = []

        # nn config 
        self.inpDim = X.shape[1]
        self.nNeuron = nNeuron 
        self.device = device 

        # neural network
        self.Alpha = None
        self.Beta = np.zeros((self.nNeuron,1)) 
        self.C = np.zeros((self.nNeuron, self.inpDim))

        # put to device 
        self.to_device()

    def to_device(self):
        self.X = torch.from_numpy(self.X) #.float()
        self.y = torch.from_numpy(self.y)#.float()
        self.yk = torch.from_numpy(self.yk)#.float()
        self.Xtest = torch.from_numpy(self.Xtest)#.float()
        self.ytest = torch.from_numpy(self.ytest)#.float()
        self.C = torch.from_numpy(self.C)#.float()
        self.Beta = torch.from_numpy(self.Beta)#.float()

        self.X = self.X.to(self.device)
        self.y = self.y.to(self.device)
        self.yk = self.yk.to(self.device)
        self.Xtest = self.Xtest.to(self.device)
        self.ytest = self.ytest.to(self.device)
        self.Beta = self.Beta.to(self.device)
        self.C = self.C.to(self.device)

    def random_guess(self, nr):
        self.betas = -1/(torch.rand(nr,1)).to(self.device)
        self.Cs = torch.rand(nr, self.inpDim).to(self.device)
        self.gs = torch.exp(self.betas * torch.cdist(self.Cs, self.X))
 
    def brute_search(self):
        # init search
        rG = self.h * self.gs @ (self.y - self.yk)
        E = -0.5 * rG ** 2 
        
        idx = torch.argmin(E)

        betak = self.betas[idx]
        Ck = self.Cs[idx]
        return betak, Ck
    
    def projection(self, k):            
        gsub = torch.exp(self.Beta[:k+1] * torch.cdist(self.C[:k+1], self.X))
         
        A = self.h * torch.einsum('kn,pn->kp', gsub, gsub)
        b = self.h * torch.einsum('kn,ns->ks', gsub, self.y)
        alpha_k = torch.linalg.solve(A, b)
        return alpha_k
  
    def optimize_random(self, nr=1024):

        for k in range(self.nNeuron):
            self.random_guess(nr)
            betak, Ck = self.brute_search()
            self.Beta[k] = betak
            self.C[k] = Ck

            try:
                alpha_k = self.projection(k)
                self.Alpha = alpha_k
            except:
                print("LinAlgWarning, we should stop adding neurons")
                break 

            # update Gk
            self.yk = (self.Alpha.T @ torch.exp(self.Beta[:k+1] * torch.cdist(self.C[:k+1], self.X))).T
            yrl2 = relative_err(self.yk.cpu().numpy(), self.y.cpu().numpy())

            ypred = (self.Alpha.T @ torch.exp(self.Beta[:k+1] * torch.cdist(self.C[:k+1], self.Xtest))).T
            test_rl2 = relative_err(ypred.cpu().numpy(), self.ytest.cpu().numpy())
            self.ylog.append(test_rl2)

            if k % 1 == 0:
                print('{:}th (y) : train rl2 {:.4e} - val rl2 : {:.4e}'.format(k, yrl2, test_rl2))

        self.model_weights = {"C":self.C, "Beta":self.Beta, "Alpha":self.Alpha}
        self.log = {"yrl2" : np.array(self.ylog)}
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGA for function fitting')
    parser.add_argument('--task', type=str, default='helmholtz2D',
                        help='dataset name. (poisson2D, helmholtz2D)')
    parser.add_argument('--nNeuron', type=int, default=257,
                        help='maximum number of neurons')
    parser.add_argument('--random', action='store_true',
                        help='flag for WOGA')
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}')
    
    if args.task == 'poisson1d':
        from utils import load_poisson1d_fitting_dataset
        X, y, Xtest, ytest = load_poisson1d_fitting_dataset()
    elif args.task == 'poisson2d':
        from utils import load_poisson2d_fitting_dataset
        X, y, Xtest, ytest = load_poisson2d_fitting_dataset(nTrain=50000)
    elif args.task == 'rbf2d':
        from utils import load_rbf2d_fitting_dataset
        X, y, Xtest, ytest = load_rbf2d_fitting_dataset()
    elif args.task.startswith('rbf1d'):
        L = int(args.task.split('-')[1])
        from utils import load_rbf1d_fitting_dataset
        X, y, Xtest, ytest = load_rbf1d_fitting_dataset(L=L)

    model = ShallowOGAFitter(
        nNeuron=args.nNeuron, X=X, y=y, 
        Xtest=Xtest, ytest=ytest, device=device)

    model.optimize_random(nr=1024)

    log_outpath, upred_outpath, model_outpath, Gpred_outpath = init_records('./results', args.task, 'rbf-oga-{:}'.format(args.nNeuron))
    np.save(log_outpath, model.log)
    # np.save(model_outpath, model.model_weights)
