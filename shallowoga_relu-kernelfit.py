import os 
import scipy 
import numpy as np
from network import ReLUk
import argparse
import torch 
torch.set_default_dtype(torch.float64)

from utils import relative_err
from utils import init_records

class ShallowOGAFitter:
    def __init__(
            self, nNeuron, act, 
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
        self.X = np.c_[X, np.ones((X.shape[0],1))]
        self.fs = fs 
        self.us = us 
        self.fTest = fTest 
        self.uTest = uTest
        self.Gref = Gref

        self.Gk = np.zeros((self.nxPts, self.nyPts))

        # neural network
        self.nNeuron = nNeuron
        self.act = act
        self.Alpha = None # np.zeros((nNeuron, 1))
        self.WB = np.zeros((nNeuron, self.inpDim + 1))

        # log 
        self.Glog = []
        self.utest_log = []
        self.utrain_log = []
        

        # put to device 
        self.device = device 
        self.to_device()
    
    def to_device(self):
        self.X = torch.from_numpy(self.X)#.float()
        self.fs = torch.from_numpy(self.fs)#.float()
        self.us = torch.from_numpy(self.us)#.float()
        self.fTest = torch.from_numpy(self.fTest)#.float()
        self.uTest = torch.from_numpy(self.uTest)#.float()
        self.WB = torch.from_numpy(self.WB)#.float()
        self.Gk = torch.from_numpy(self.Gk)#.float()

        self.X = self.X.to(self.device)
        self.fs = self.fs.to(self.device)
        self.us = self.us.to(self.device)
        self.fTest = self.fTest.to(self.device)
        self.uTest = self.uTest.to(self.device)
        self.WB = self.WB.to(self.device)
        self.Gk = self.Gk.to(self.device)

    def random_guess(self, nr):

        if self.inpDim == 2:
            phi = torch.rand(nr).to(self.device) * 2*torch.pi
            b = (torch.rand(nr)*2 - 1).to(self.device) * 2**1.5  # for poisson 1D
            Wx = torch.cos(phi).reshape(-1,1)
            Wy = torch.sin(phi).reshape(-1,1)
            B = b.reshape(-1,1)
            self.WBs = torch.concatenate([Wx, Wy, B], axis=1)  
            self.nParam = self.WBs.shape[0]

        if self.inpDim == 3:
            phi1 = torch.rand(nr).to(self.device) * torch.pi 
            phi2 = torch.rand(nr).to(self.device) * 2 * torch.pi 
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
            b = (torch.rand(nr)*2 - 1.0).to(self.device) * 8 # for poisson 2D
            
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
        
        if self.inpDim == 6:
            phi1 = torch.rand(nr).to(self.device) * torch.pi 
            phi2 = torch.rand(nr).to(self.device) * torch.pi 
            phi3 = torch.rand(nr).to(self.device) * torch.pi 
            phi4 = torch.rand(nr).to(self.device) * torch.pi 
            phi5 = torch.rand(nr).to(self.device) * 2 * torch.pi 
            b = (torch.rand(nr)*2 - 1.0).to(self.device) * 8 # for poisson 2D
            
            w1 = torch.cos(phi1)
            w2 = torch.sin(phi1) * torch.cos(phi2)
            w3 = torch.sin(phi1) * torch.sin(phi2) * torch.cos(phi3)
            w4 = torch.sin(phi1) * torch.sin(phi2) * torch.sin(phi3) * torch.cos(phi4)
            w5 = torch.sin(phi1) * torch.sin(phi2) * torch.sin(phi3) * torch.sin(phi4) * torch.cos(phi5)
            w6 = torch.sin(phi1) * torch.sin(phi2) * torch.sin(phi3) * torch.sin(phi4) * torch.sin(phi5)
                        
            w1 = w1.reshape(-1,1)
            w2 = w2.reshape(-1,1)
            w3 = w3.reshape(-1,1)
            w4 = w4.reshape(-1,1)
            w5 = w5.reshape(-1,1)
            w6 = w6.reshape(-1,1)
            
            b = b.reshape(-1,1)
            self.WBs = torch.concatenate([w1, w2, w3, w4, w5, w6, b], axis=1)   
            self.nParam = self.WBs.shape[0]

        gs = self.act(self.WBs @ self.X.T) # nParam x (nxPts x nyPts)
        self.gs = gs.reshape(self.nParam, self.nxPts, self.nyPts)

    def brute_search(self):
        uts = self.h * self.Gk @ self.fs 
        uhss = self.h * self.gs @ self.fs
        rG = self.h * ((self.us - uts) * uhss).sum(axis=(1,2))
        E = -0.5 * rG ** 2 
        
        idx = torch.argmin(E)

        wbk = self.WBs[idx]
        return wbk 

    def projection(self, k):
        gsub = self.act(self.WB[:k+1] @ self.X.T)
        gsub = gsub.reshape(k+1, self.nxPts, self.nyPts)
        uhsub = self.h * gsub @ self.fs

        A = self.h * torch.einsum('kns,pns->kp', uhsub, uhsub)
        b = self.h * torch.einsum('kns,ns->k', uhsub, self.us)[...,None]
        alpha_k = torch.linalg.solve(A, b)
        return alpha_k.reshape(-1)

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
            
            # update Gk
            gs = self.act(self.WB[:k+1] @ self.X.T)
            gs = gs.reshape(k+1, self.nxPts, self.nyPts)
            self.Gk = torch.einsum('k,kxy->xy', self.Alpha, gs)

            if self.Gref is not None:
                if np.isnan(self.Gref).sum()>0:
                    Gref = self.Gref.reshape(-1)
                    Gmask = ~np.isnan(Gref) & ~np.isinf(Gref)
                    Grl2 = relative_err(self.Gk.cpu().reshape(-1,1).numpy()[Gmask], Gref[Gmask])
                else:
                    Grl2 = relative_err(self.Gk.cpu().reshape(-1,1).numpy(), self.Gref.reshape(-1,1))
                self.Glog.append(Grl2)
            else:
                Grl2 = 1.0

            utrain_Pred = self.h * self.Gk @ self.fs
            utest_Pred = self.h * self.Gk @ self.fTest

            utrain_rl2 = relative_err(utrain_Pred.T.cpu().numpy(), self.us.T.cpu().numpy())
            utest_rl2 = relative_err(utest_Pred.T.cpu().numpy(), self.uTest.T.cpu().numpy())

            self.utest_log.append(utest_rl2)
            self.utrain_log.append(utrain_rl2)
            

            if k % 1 == 0:
                print('{:}th train url2 : {:.4e} - test url2 : {:.4e} - Grl2 : {:.4e}'.format(k, utrain_rl2, utest_rl2, Grl2))
        
        print('{:}th url2 : {:.4e} - Grl2 : {:.4e}'.format(k, utest_rl2, Grl2))        

        self.Gk = {"X": self.X.cpu().numpy(), "Gpred" : self.Gk.cpu().numpy(), "Gref":self.Gref}
        self.utest_Pred = utest_Pred.cpu().numpy()
        self.model_weights = {"WB":self.WB.detach().cpu().numpy(), "Alpha":self.Alpha.detach().cpu().numpy()}
        self.log = {"G_rl2" : np.array(self.Glog), "utest_rl2" : np.array(self.utest_log), "utrain_rl2" : np.array(self.utrain_log)}
        # print(self.Alpha)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGA for kernel estimation')
    parser.add_argument('--task', type=str, default='poisson1D',
                        help='dataset name. (poisson1D, helmholtz1D, airy1D, poisson2D)')
    parser.add_argument('--act', type=str, default='relu',
                        help='activiation name. (relu, tanh, relu2, sigmoid)')
    parser.add_argument('--nNeuron', type=int, default=257, 
                        help='maximum number of neurons')
    parser.add_argument('--nTrain', type=int, default=10, 
                        help='number of training samples')
    parser.add_argument('--nTest', type=int, default=200, 
                        help='number of test samples')
    parser.add_argument('--sigma', type=str, default='2e-1', 
                        help='number of test samples')
    parser.add_argument('--res', type=int, default=20, 
                        help='mesh density')
    parser.add_argument('--nr', type=int, default=512, 
                        help='mesh density')
    parser.add_argument('--param', type=float, default=1,
                        help='mesh density')
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
    elif args.task == 'cos2D':
        from utils import load_cos2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_cos2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, n=args.param, res=args.res, sigma=args.sigma)
    elif args.task == 'gauss2D':
        from utils import load_gauss2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_gauss2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, c=args.param, res=args.res, sigma=args.sigma)
    elif args.task == 'poisson2D':
        from utils import load_poisson2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_poisson2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=args.res, sigma=args.sigma)
        args.param = 'x'
    elif args.task == 'helmholtz2D':
        from utils import load_helmholtz2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmholtz2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=args.res, sigma=args.sigma)
        args.param = 'x'
    elif args.task == 'cos2Dhdomain':
        from utils import load_cos2dhdomain_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_cos2dhdomain_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, n=1)
    elif args.task == 'poisson2Dhdomain':
        from utils import load_poisson2dhdomain_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_poisson2dhdomain_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest,res=args.res, sigma=args.sigma)
        args.param = 'x'
    elif args.task == 'helmholtz2Dhdomain':
        from utils import load_helmholtz2dhdomain_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmholtz2dhdomain_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest,res=args.res, sigma=args.sigma)
        args.param = 'x'


    act = ReLUk(k=1)
    model = ShallowOGAFitter(
        nNeuron=args.nNeuron, act=act, X=X, 
        fs=fTrain, us=uTrain, 
        fTest=fTest, uTest=uTest, 
        Gref=Gref, device=device)
    
    exp_nm = 'oga-{:}-{:}-{:}-{:}-{:}-{:}-relu'.format(args.nNeuron, args.nTrain, args.nr, args.res, args.param, args.sigma)
    print(exp_nm)
    log_outpath, upred_outpath, model_outpath, Gpred_outpath = init_records(
        './results', args.task, exp_nm)
    if os.path.exists(upred_outpath):
        print("********* ", args.task, "  ", exp_nm, " file exists")
        exit()

    # model train
    model.optimize_random(nr=args.nr)

    # # save outputs
    # if args.nr == 512 & args.res == 20:
    #     log_outpath, upred_outpath, model_outpath, Gpred_outpath = init_records('./results', args.task, 'oga-{:}-{:}-relu'.format(args.nNeuron, args.nTrain))
    # else:

    

    np.save(log_outpath, model.log)
    np.save(upred_outpath, model.utest_Pred)
    np.save(Gpred_outpath, model.Gk)
    np.save(model_outpath, model.model_weights)