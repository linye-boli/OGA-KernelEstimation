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
            nNeuron, act, 
            X, y, Xtest, ytest,
            device):

        if X.shape[1] == 4:
            Rx = (X[:,[0]]**2 + X[:,[1]]**2) ** 0.5
            Ry = (X[:,[2]]**2 + X[:,[3]]**2) ** 0.5
            Dxy = ((X[:,[0]] - X[:,[2]])**2 + (X[:,[1]]-X[:,[3]])**2) ** 0.5
            X = np.c_[Rx, Ry, Dxy]

            Rx = (Xtest[:,[0]]**2 + Xtest[:,[1]]**2) ** 0.5
            Ry = (Xtest[:,[2]]**2 + Xtest[:,[3]]**2) ** 0.5
            Dxy = ((Xtest[:,[0]] - Xtest[:,[2]])**2 + (Xtest[:,[1]]-Xtest[:,[3]])**2) ** 0.5
            Xtest = np.c_[Rx, Ry, Dxy]
        elif X.shape[1] == 2:
            Rx = X[:,[0]]
            Ry = X[:,[1]]
            Dxy = np.abs(X[:,[0]] - X[:,[1]])
            X = np.c_[Rx, Ry, Dxy]

            Rx = Xtest[:,[0]]
            Ry = Xtest[:,[1]]
            Dxy = np.abs(Xtest[:,[0]] - Xtest[:,[1]])
            Xtest = np.c_[Rx, Ry, Dxy]

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

        # put to device 
        self.device = device 
        self.to_device()

    def to_device(self):
        self.X = torch.from_numpy(self.X) #.float()
        self.y = torch.from_numpy(self.y) #.float()
        self.yk = torch.from_numpy(self.yk) #.float()
        self.Xtest = torch.from_numpy(self.Xtest) #.float()
        self.ytest = torch.from_numpy(self.ytest) #.float()
        self.WB = torch.from_numpy(self.WB) #.float()

        self.X = self.X.to(self.device)
        self.y = self.y.to(self.device)
        self.yk = self.yk.to(self.device)
        self.Xtest = self.Xtest.to(self.device)
        self.ytest = self.ytest.to(self.device)
        self.WB = self.WB.to(self.device)


    def brute_guess(self, nw, nb):

        if self.inpDim == 1:
            phi = torch.linspace(0, torch.pi, nw).to(self.device)
            b = torch.linspace(-2**1.5, 2**1.5, nb).to(self.device)
            phi, b = torch.meshgrid((phi, b), indexing='ij')
            Wx = torch.cos(phi).reshape(-1,1)
            B = b.reshape(-1,1)
            self.WBs = torch.concatenate([Wx, B], axis=1)   
            self.nParam = self.WBs.shape[0]

        if self.inpDim == 2:
            phi = torch.linspace(0, torch.pi, nw).to(self.device)
            b = torch.linspace(-2**1.5, 2**1.5, nb).to(self.device)
            phi, b = torch.meshgrid((phi, b), indexing='ij')
            Wx = torch.cos(phi).reshape(-1,1)
            Wy = torch.sin(phi).reshape(-1,1)
            B = b.reshape(-1,1)
            self.WBs = torch.concatenate([Wx, Wy, B], axis=1)   
            self.nParam = self.WBs.shape[0]
        
        if self.inpDim == 4:
            phi1 = torch.linspace(0, torch.pi, nw).to(self.device)
            phi2 = torch.linspace(0, torch.pi, nw).to(self.device)
            phi3 = torch.linspace(0, 2*torch.pi, nw).to(self.device)
            b = torch.linspace(-2,2,nb).to(self.device)
            phi1, phi2, phi3, b = torch.meshgrid(
                (phi1, phi2, phi3, b), indexing='ij')
            w1 = torch.cos(phi1)
            w2 = torch.sin(phi1) * torch.cos(phi2)
            w3 = torch.sin(phi1) * torch.sin(phi2) * torch.cos(phi3)
            w4 = torch.sin(phi1) * torch.sin(phi2) * torch.sin(phi3)
            w1 = w1.reshape(-1,1)
            w2 = w2.reshape(-1,1)
            w3 = w3.reshape(-1,1)
            w4 = w4.reshape(-1,1)
            b = b.reshape(-1,1)
            self.WBs = torch.concatenate([w1, w2, w3, w4, b], axis=1)   
            self.nParam = self.WBs.shape[0]
         
        self.gs = self.act(self.WBs @ self.X.T) 

        print()
        print('#{:} basis calculation finish'.format(self.nParam))
        print()

    def random_guess(self, nr):
        if self.inpDim == 1:
            phi = torch.rand(nr).to(self.device) * 2*torch.pi
            b = (torch.rand(nr)*2 - 1).to(self.device) * 2**1.5
            Wx = torch.cos(phi).reshape(-1,1)
            B = b.reshape(-1,1)
            self.WBs = torch.concatenate([Wx, B], axis=1)  
            self.nParam = self.WBs.shape[0]

        if self.inpDim == 2:
            phi = torch.rand(nr).to(self.device) * 2*torch.pi
            b = (torch.rand(nr)*2 - 1).to(self.device) * 2**1.5
            Wx = torch.cos(phi).reshape(-1,1)
            Wy = torch.sin(phi).reshape(-1,1)
            B = b.reshape(-1,1)
            self.WBs = torch.concatenate([Wx, Wy, B], axis=1)  
            self.nParam = self.WBs.shape[0]

        if self.inpDim == 3:
            phi1 = torch.rand(nr).to(self.device) * torch.pi 
            phi2 = torch.rand(nr).to(self.device) *  2 * torch.pi 
            b = (torch.rand(nr)*2 - 1.0).to(self.device) * 4 # for poisson 2D
            
            w1 = torch.cos(phi1)
            w2 = torch.sin(phi1) * torch.cos(phi2)
            w3 = torch.sin(phi1) * torch.sin(phi2)
            w1 = w1.reshape(-1,1)
            w2 = w2.reshape(-1,1)
            w3 = w3.reshape(-1,1)
            b = b.reshape(-1,1)
            self.WBs = torch.concatenate([w1, w2, w3, b], axis=1)   
            self.nParam = self.WBs.shape[0]

        if self.inpDim == 4:
            phi1 = torch.rand(nr).to(self.device) * torch.pi 
            phi2 = torch.rand(nr).to(self.device) * torch.pi 
            phi3 = torch.rand(nr).to(self.device) * 2 * torch.pi 
            b = (torch.rand(nr)*2 - 1.0).to(self.device) * 4 # for poisson 2D
            
            w1 = torch.cos(phi1)
            w2 = torch.sin(phi1) * torch.cos(phi2)
            w3 = torch.sin(phi1) * torch.sin(phi2) * torch.cos(phi3)
            w4 = torch.sin(phi1) * torch.sin(phi2) * torch.sin(phi3)
            w1 = w1.reshape(-1,1)
            w2 = w2.reshape(-1,1)
            w3 = w3.reshape(-1,1)
            w4 = w4.reshape(-1,1)
            b = b.reshape(-1,1)
            self.WBs = torch.concatenate([w1, w2, w3, w4, b], axis=1)   
            self.nParam = self.WBs.shape[0]

        
        self.gs = self.act(self.WBs @ self.X.T)
 
    def brute_search(self):
        # init search
        rG = self.h * self.gs @ (self.y - self.yk)
        E = -0.5 * rG ** 2 
        
        idx = torch.argmin(E)

        wbk = self.WBs[idx]
        return wbk
    
    def projection(self, k):    
        gsub = self.act(self.WB[:k+1] @ self.X.T)
        A = self.h * torch.einsum('kn,pn->kp', gsub, gsub)
        b = self.h * torch.einsum('kn,ns->ks', gsub, self.y)

        # import pdb 
        # pdb.set_trace()

        alpha_k = torch.linalg.solve(A, b)
        return alpha_k

    def optimize_deterministic(self, nw=4, nb=101):
        self.brute_guess(nw, nb)

        for k in range(self.nNeuron):

            wbk = self.brute_search()
            self.WB[k] = wbk

            try:
                alpha_k = self.projection(k)
                self.Alpha = alpha_k
            except:
                print("LinAlgWarning, we should stop adding neurons")
                break 

            # update Gk
            self.yk = (self.Alpha.T @ self.act(self.WB[:k+1] @ self.X.T)).T

            yrl2 = relative_err(self.yk.cpu().numpy(), self.y.cpu().numpy())
            # self.ylog.append(yrl2)
            
            ypred = (self.Alpha.T @ self.act(self.WB[:k+1] @ self.Xtest.T)).T
            test_rl2 = relative_err(ypred.cpu().numpy(), self.ytest.cpu().numpy())
            self.ylog.append(test_rl2)
            if k % 5 == 0:
                print('{:}th (y) : train rl2 {:.4e} - val rl2 : {:.4e}'.format(k, yrl2, test_rl2))

        # self.model_weights = {"WB":self.WB, "Alpha":self.Alpha}
        self.log = {"yrl2" : np.array(self.ylog)}
    
    def optimize_random(self, nr=1024):

        for k in range(self.nNeuron):
            self.random_guess(nr)
            wbk = self.brute_search()
            self.WB[k] = wbk

            try:
                alpha_k = self.projection(k)
                self.Alpha = alpha_k
            except:
                print("LinAlgWarning, we should stop adding neurons")
                break 

            # update yk
            self.yk = (self.Alpha.T @ self.act(self.WB[:k+1] @ self.X.T)).T         
            
            yrl2 = relative_err(self.yk.cpu().numpy(), self.y.cpu().numpy())
            ymse = np.mean((self.yk.cpu().numpy() - self.y.cpu().numpy())**2)

            ypred = (self.Alpha.T @ self.act(self.WB[:k+1] @ self.Xtest.T)).T
            test_rl2 = relative_err(ypred.cpu().numpy(), self.ytest.cpu().numpy())
            test_mse = np.mean((ypred.cpu().numpy() - self.ytest.cpu().numpy())**2)
            self.ylog.append(test_rl2)

            if k % 100 == 0:
                print('{:}th (y) : train rl2 {:.4e} - val rl2 : {:.4e}'.format(
                    k, yrl2, test_rl2))

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
    parser.add_argument('--k', type=int, default=1,
                        help='order of relu')
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
    elif args.task == 'log1d':
        from utils import load_log1d_fitting_dataset
        X, y, Xtest, ytest = load_log1d_fitting_dataset()
    elif args.task == 'log2d':
        from utils import load_log2d_fitting_dataset
        X, y, Xtest, ytest = load_log2d_fitting_dataset()
    elif args.task == 'localoscillatory1d':
        from utils import load_localoscillatory1d_fitting_dataset
        X, y, Xtest, ytest = load_localoscillatory1d_fitting_dataset()
    
    act = ReLUk(k=args.k)
    model = ShallowOGAFitter(
        nNeuron=args.nNeuron, act=act, X=X, y=y, 
        Xtest=Xtest, ytest=ytest, device=device)

    model.optimize_random(nr=2048)

    log_outpath, upred_outpath, model_outpath, Gpred_outpath = init_records('./results', args.task, 'relu{:}-oga-{:}'.format(args.k, args.nNeuron))
    np.save(log_outpath, model.log)
    # np.save(model_outpath, model.model_weights)
