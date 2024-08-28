import os 
import scipy 
import numpy as np
from network import ReLUk, Hat, Gauss
import argparse
import torch 
# torch.set_default_dtype(torch.float64)

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
            # Rx = (X[:,[0]]**2 + X[:,[1]]**2) ** 0.5
            # Ry = (X[:,[2]]**2 + X[:,[3]]**2) ** 0.5
            # Dxy = ((X[:,[0]] - X[:,[2]])**2 + (X[:,[1]]-X[:,[3]])**2) ** 0.5
            # X = np.c_[Rx, Ry, Dxy]
            area = np.pi 
        elif X.shape[1] == 2:
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
        self.C = np.zeros((self.nNeuron, 1))

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
        self.WB = torch.from_numpy(self.WB).float()
        self.Gk = torch.from_numpy(self.Gk).float()
        self.C = torch.from_numpy(self.C).float()

        self.X = self.X.to(self.device)
        self.fs = self.fs.to(self.device)
        self.us = self.us.to(self.device)
        self.fTest = self.fTest.to(self.device)
        self.uTest = self.uTest.to(self.device)
        self.WB = self.WB.to(self.device)
        self.Gk = self.Gk.to(self.device)
        self.C = self.C.to(self.device)

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
            # phi1 = torch.rand(nr).to(self.device) * torch.pi 
            # phi2 = torch.rand(nr).to(self.device) *  2 * torch.pi 
            # b = (torch.rand(nr)*2 - 1.0).to(self.device) * 4 # for poisson 2D
            
            # w1 = torch.cos(phi1)
            # w2 = torch.sin(phi1) * torch.cos(phi2)
            # w3 = torch.sin(phi1) * torch.sin(phi2)
            # w1 = w1.reshape(-1,1)
            # w2 = w2.reshape(-1,1)
            # w3 = w3.reshape(-1,1)
            # b = b.reshape(-1,1)
            # self.WBs = torch.concatenate([w1, w2, w3, b], axis=1)   
            # self.nParam = self.WBs.shape[0]

            w = torch.rand(nr,3) * 2 - 1
            b = (torch.rand(nr,1) * 2 - 1)*4
            c = 1/torch.rand(nr,1) 
            self.WBs = torch.concatenate([w, b], axis=1).to(device)
            self.Cs = c.to(device)
            self.nParam = self.WBs.shape[0]
        
        if self.inpDim == 4:
            w = torch.randn(nr,4)
            b = torch.randn(nr,1) 
            c = torch.ones(nr,1) * 0.5
            self.WBs = torch.concatenate([w, b], axis=1).to(device)
            self.Cs = c.to(device)
            self.nParam = self.WBs.shape[0]
        
        gs = self.act(self.WBs @ self.X.T, self.Cs)
        self.gs = gs.reshape(self.nParam, self.nxPts, self.nyPts)

    def brute_search(self):
        uts = self.h * self.Gk @ self.fs
        uhss = self.h * self.gs @ self.fs
        rG = self.h * ((self.us - uts) * uhss).sum(axis=(1,2))
        E = -0.5 * rG ** 2 
        
        idx = torch.argmin(E)

        wbk = self.WBs[idx]
        ck = self.Cs[idx]
        return wbk, ck

    def projection(self, k):
        gsub = self.act(self.WB[:k+1] @ self.X.T, self.C[:k+1])
        gsub = gsub.reshape(k+1, self.nxPts, self.nyPts)
        uhsub = self.h * gsub @ self.fs

        A = self.h * torch.einsum('kns,pns->kp', uhsub, uhsub)
        b = self.h * torch.einsum('kns,ns->k', uhsub, self.us)[...,None]
        alpha_k = torch.linalg.solve(A, b)
        return alpha_k.reshape(-1)

    def optimize_random(self, nr=1024):
        for k in range(self.nNeuron):
            self.random_guess(nr)
            wbk, ck = self.brute_search()
            self.WB[k] = wbk
            self.C[k] = ck

            # try:
            alpha_k = self.projection(k)
            self.Alpha = alpha_k
            # except:
            #     print("LinAlgWarning, we should stop adding neurons")
            #     break 
            
            
            # update Gk
            gs = self.act(self.WB[:k+1] @ self.X.T, self.C[:k+1])
            gs = gs.reshape(k+1, self.nxPts, self.nyPts)
            self.Gk = torch.einsum('k,kxy->xy', self.Alpha, gs)

            if self.Gref is not None:
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
               
        # print('{:}th url2 : {:.4e} - Grl2 : {:.4e}'.format(k, url2, Grl2))        
        self.Gk = self.Gk.cpu().numpy()
        self.utest_Pred = utest_Pred.cpu().numpy()
        self.model_weights = {"WB":self.WB.cpu().numpy(), "Alpha":self.Alpha.cpu().numpy(), "C":self.C.cpu().numpy()}
        self.log = {"G_rl2" : np.array(self.Glog), "utest_rl2" : np.array(self.utest_log), "utrain_rl2" : np.array(self.utrain_log)}
    
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
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}')
    # device = torch.device('cpu')
    
    
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

    
    act = Gauss()
    model = ShallowOGAFitter(
        nNeuron=args.nNeuron, act=act, X=X, 
        fs=fTrain, us=uTrain, 
        fTest=fTest, uTest=uTest, 
        Gref=Gref, device=device)
    # model train
    model.optimize_random(nr=256)

    # # save outputs
    log_outpath, upred_outpath, model_outpath, Gpred_outpath = init_records('./results', args.task, 'oga-{:}-{:}-hat'.format(args.nNeuron, args.nTrain))
    np.save(log_outpath, model.log)
    np.save(upred_outpath, model.utest_Pred)
    np.save(Gpred_outpath, model.Gk)
    np.save(model_outpath, model.model_weights)