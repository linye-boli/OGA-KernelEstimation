import argparse
import numpy as np
import torch 
torch.set_default_dtype(torch.float64)

from utils import relative_err
from utils import init_records
from network import Sine

class ShallowOGAFitter:
    def __init__(
            self, 
            nNeuron, act, 
            X, y, Xtest, ytest,
            device):
        
        # dataset 
        self.nPts = X.shape[0]
        self.inpDim = X.shape[1]
        self.X = np.c_[X, np.ones((X.shape[0],1))]
        self.y = y
        self.Xtest = np.c_[Xtest, np.ones((Xtest.shape[0],1))]
        self.ytest = ytest
        self.h = 1/self.nPts
        self.ylog = []

        self.yk = np.zeros_like(y)

        # nn config 
        self.nNeuron = nNeuron 
        self.act = act 
        self.Alpha = None
        self.WB = np.zeros((self.nNeuron, self.inpDim+1))
        self.C = np.zeros((self.nNeuron,1)) 
        self.M = np.zeros((self.nNeuron, self.inpDim))

        # put to device 
        self.device = device 
        self.to_device()

    def to_device(self):
        self.X = torch.from_numpy(self.X) #.float()
        self.y = torch.from_numpy(self.y)#.float()
        self.yk = torch.from_numpy(self.yk)#.float()
        self.Xtest = torch.from_numpy(self.Xtest)#.float()
        self.ytest = torch.from_numpy(self.ytest)#.float()
        self.WB = torch.from_numpy(self.WB)#.float()
        self.C = torch.from_numpy(self.C)#.float()
        self.M = torch.from_numpy(self.M)#.float()

        self.X = self.X.to(self.device)
        self.y = self.y.to(self.device)
        self.yk = self.yk.to(self.device)
        self.Xtest = self.Xtest.to(self.device)
        self.ytest = self.ytest.to(self.device)
        self.WB = self.WB.to(self.device)
        self.M = self.M.to(self.device)
        self.C = self.C.to(self.device)

    def random_guess(self, nr, k=1000):
        if self.inpDim == 1:
            w = (torch.rand(nr,1) * 2 - 1) * k * torch.pi
            b = (torch.rand(nr,1) * 2 - 1)*torch.pi
            self.WBs = torch.concatenate([w, b], axis=1).to(device)
            self.nParam = self.WBs.shape[0]

        if self.inpDim == 2:
            w = (torch.rand(nr,2) * 2 - 1) * k * torch.pi
            b = (torch.rand(nr,1) * 2 - 1)*torch.pi
            
            self.Ms = torch.rand(nr, 2).to(device)
            self.Cs = -1/ torch.rand(nr, 1).to(device)
            self.WBs = torch.concatenate([w, b], axis=1).to(device)
            self.nParam = self.WBs.shape[0]
        
        if self.inpDim == 4:
            w = (torch.rand(nr,4) * 2 - 1) * k * torch.pi
            b = (torch.rand(nr,1) * 2 - 1)*torch.pi
            
            self.Ms = (torch.rand(nr, 4)*2 - 1).to(device)
            self.Cs = - 1/ torch.rand(nr, 1).to(device)
            self.WBs = torch.concatenate([w, b], axis=1).to(device)
            self.nParam = self.WBs.shape[0]
        
        self.gs = self.act(self.WBs @ self.X.T) * torch.exp(self.Cs * torch.cdist(self.Ms, self.X[:,:-1]))
 
    def brute_search(self):
        # init search
        rG = self.h * self.gs @ (self.y - self.yk)
        E = -0.5 * rG ** 2

        idx = torch.argmin(E)

        wbk = self.WBs[idx]
        ck = self.Cs[idx]
        mk = self.Ms[idx]
        return wbk, ck, mk
    
    def projection(self, k):
        gsub = self.act(self.WB[:k+1] @ self.X.T) * torch.exp(self.C[:k+1] * torch.cdist(self.M[:k+1], self.X[:,:-1]))
        A = self.h * torch.einsum('kn,pn->kp', gsub, gsub)
        b = self.h * torch.einsum('kn,ns->ks', gsub, self.y)

        # import pdb 
        # pdb.set_trace()

        alpha_k = torch.linalg.solve(A, b)
        return alpha_k
   
    def optimize_random(self, nr=1024):

        for k in range(self.nNeuron):
            self.random_guess(nr, 70)
            wbk, ck, mk = self.brute_search()

            self.WB[k] = wbk
            self.C[k] = ck 
            self.M[k] = mk

            try:
                alpha_k = self.projection(k)
                self.Alpha = alpha_k
            except:
                print("LinAlgWarning, we should stop adding neurons")
                break 

            # update yk
            gs = self.act(self.WB[:k+1] @ self.X.T) * torch.exp(self.C[:k+1] * torch.cdist(self.M[:k+1], self.X[:,:-1]))
            self.yk = (self.Alpha.T @ gs).T         
            
            yrl2 = relative_err(self.yk.cpu().numpy(), self.y.cpu().numpy())
            ymse = np.mean((self.yk.cpu().numpy() - self.y.cpu().numpy())**2)

            gs = self.act(self.WB[:k+1] @ self.Xtest.T) * torch.exp(self.C[:k+1] * torch.cdist(self.M[:k+1], self.Xtest[:,:-1]))
            ypred = (self.Alpha.T @ gs).T
            test_rl2 = relative_err(ypred.cpu().numpy(), self.ytest.cpu().numpy())
            test_mse = np.mean((ypred.cpu().numpy() - self.ytest.cpu().numpy())**2)
            self.ylog.append(test_rl2)

            if k % 100 == 0:
                print('{:}th (y) : train rl2 {:.4e} - val rl2 : {:.4e}'.format(k, yrl2, test_rl2))
                # print('{:}th (y) : train rl2 {:.4e} - train mse {:.4e} - val rl2 : {:.4e} - val mse : {:.4e}'.format(
                #     k, yrl2, np.log10(ymse), test_rl2, np.log10(test_mse)))

        print('{:}th (y) : train rl2 {:.4e} - train mse {:.4e} - val rl2 : {:.4e} - val mse : {:.4e}'.format(
                    k, yrl2, np.log10(ymse), test_rl2, np.log10(test_mse)))
        self.model_weights = {"WB":self.WB, "Alpha":self.Alpha}
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
        XN = np.linalg.norm(X[:,:2] - X[:,2:], axis=1) 
        XtestN = np.linalg.norm(Xtest[:,:2] - Xtest[:,2:], axis=1)
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
    elif args.task == 'oscillatory1d':
        from utils import load_oscillatory1d_fitting_dataset
        X, y, Xtest, ytest = load_oscillatory1d_fitting_dataset()
    elif args.task == 'arctan1d':
        from utils import load_oscillatory1d_fitting_dataset
        X, y, Xtest, ytest = load_oscillatory1d_fitting_dataset()
    elif args.task == 'localoscillatory1d':
        from utils import load_localoscillatory1d_fitting_dataset
        X, y, Xtest, ytest = load_localoscillatory1d_fitting_dataset()
    


    act = Sine()
    model = ShallowOGAFitter(
        nNeuron=args.nNeuron, act=act, X=X, y=y, 
        Xtest=Xtest, ytest=ytest, device=device)

    if args.random:
        model.optimize_random(nr=2048)
    else:
        model.optimize_deterministic(nw=3,nb=101)

    log_outpath, upred_outpath, model_outpath, Gpred_outpath = init_records('./results', args.task, 'sin-oga-{:}'.format(args.nNeuron))
    np.save(log_outpath, model.log)
    # np.save(model_outpath, model.model_weights)
