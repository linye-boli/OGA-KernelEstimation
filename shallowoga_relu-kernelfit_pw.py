import os 
import scipy 
import numpy as np
from network import ReLUk
import argparse
import torch 
from tqdm import trange
torch.set_default_dtype(torch.float64)

from utils import relative_err
from utils import init_records
import multiprocessing as mp
from tqdm import tqdm, trange
import time 

class ShallowOGAFitter:
    def __init__(
            self, nNeuron, act, 
            X, fs, us, Idx, fTest, uTest, #Gref=None, 
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
        self.Idx = Idx
        self.nyPts = fs.shape[0]
        self.nxPts = us.shape[0]
        self.h = area/self.nyPts
        self.inpDim = X.shape[1]
        X = X.reshape(self.nxPts, self.nyPts,-1)

        # import pdb 
        # pdb.set_trace()

        if self.inpDim == 4:
            X = X[Idx,:,2:]
            # X = X[Idx]
        elif self.inpDim == 2:
            X = X[Idx,:,1:]
        elif self.inpDim == 6:
            X = X[Idx,:,3:]
        

        self.nxPts = 1
        self.inpDim = X.shape[1]

        self.X = np.c_[X, np.ones((X.shape[0],1))]
        self.fs = fs
        self.us = us[[Idx]]

        self.fTest = fTest
        self.uTest = uTest[[Idx]]

        # if Gref is not None:
        #     self.Gref = Gref[[Idx]]
        # else:
        #     self.Gref = None

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

        if self.inpDim == 1:
            b = (torch.rand(nr)*2 - 1).to(self.device) * 2  # for poisson 1D
            Wx = ((torch.rand(nr) > 0.5).float() - 0.5).to(self.device) * 2
            Wx = Wx.reshape(-1,1)
            B = b.reshape(-1,1)
            self.WBs = torch.concatenate([Wx, B], axis=1)  
            self.nParam = self.WBs.shape[0]

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

            # utrain_Pred = self.h * self.Gk @ self.fs
            # utest_Pred = self.h * self.Gk @ self.fTest

            # utrain_rl2 = relative_err(utrain_Pred.T.cpu().numpy(), self.us.T.cpu().numpy())
            # utest_rl2 = relative_err(utest_Pred.T.cpu().numpy(), self.uTest.T.cpu().numpy())

            # if k % 1 == 0:
            #     print('{:}th train url2 : {:.4e} - test url2 : {:.4e}'.format(k, utrain_rl2, utest_rl2))


        self.Gk = self.Gk.cpu().numpy()


def OGAtrain(idx, kwargs):
    kwargs['Idx'] = idx
    model = ShallowOGAFitter(**kwargs)
    model.optimize_random(nr=512)

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
    parser.add_argument('--res', type=int, default=20, 
                        help='mesh density')
    parser.add_argument('--nr', type=int, default=512, 
                        help='mesh density')
    parser.add_argument('--sigma', type=str, default='2e-1', 
                        help='number of test samples')
    parser.add_argument('--param', type=float, default=1,
                        help='mesh density')
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    parser.add_argument("--mode", type=str, default='full')
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
    elif args.task == 'cos2Dhdomain':
        from utils import load_cos2dhdomain_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_cos2dhdomain_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, n=args.param, res=args.res, sigma=args.sigma)
    elif args.task == 'poisson2D':
        from utils import load_poisson2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_poisson2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=args.res, sigma=args.sigma)
        args.param = 'x'
    elif args.task == 'poisson2Dhdomain':
        from utils import load_poisson2dhdomain_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_poisson2dhdomain_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest,res=args.res, sigma=args.sigma)
        args.param = 'x'
    elif args.task == 'helmholtz2D':
        from utils import load_helmholtz2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmholtz2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=args.res, sigma=args.sigma)
        args.param = 'x'
    elif args.task == 'helmholtz2Dhdomain':
        from utils import load_helmholtz2dhdomain_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmholtz2dhdomain_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=args.res, sigma=args.sigma)
        args.param = 'x'
    elif args.task == 'helmholtz3D':
        from utils import load_helmholtz3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmholtz3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=args.res, sigma=args.sigma)
    elif args.task == 'poisson3D':
        from utils import load_poisson3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_poisson3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=args.res, sigma=args.sigma)
    elif args.task == 'helmholtz3D':
        from utils import load_logsin3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_logsin3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'inv3D':
        from utils import load_inv3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_inv3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'log3D':
        from utils import load_log3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_log3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, res=args.res, sigma=args.sigma)
        args.param = 'x'
    elif args.task == 'logsin3D':
        from utils import load_logsin3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_logsin3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
    elif args.task == 'logcos3D':
        from utils import load_logcos3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_logcos3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, n=args.param, res=args.res, sigma=args.sigma)
    elif args.task == 'cos3D':
        from utils import load_cos3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_cos3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, n=args.param, res=args.res, sigma=args.sigma)
        
    act = ReLUk(k=1)
    # act = ReLUk(k=4)

    if X.shape[1] == 4:
        area = np.pi 
    else:
        area = 1

    nIdx = uTrain.shape[0]
    abnormals = []
    G = np.zeros((nIdx, nIdx))
    log_outpath, upred_outpath, model_outpath, Gpred_outpath = init_records('./results', args.task, 'ogapw-{:}-{:}-relu'.format(args.nNeuron, args.nTrain))

    start_time = time.time()
    for Idx in trange(nIdx):
        model = ShallowOGAFitter(
            nNeuron=args.nNeuron, act=act, X=X, 
            fs=fTrain, us=uTrain, fTest=fTest, uTest=uTest,
            Idx=Idx, device=device)
        # model train
        # if (args.task == 'log3D') | (args.task == 'logsin3D'):
        #     model.optimize_random(nr=1024)
        # else:
        model.optimize_random(nr=512)

        G[Idx] = model.Gk

        if 'cos' in args.task:
            Grl2 = relative_err(G[:Idx+1], Gref[:Idx+1])
        else:
            Grl2 = 1.0

        h = area/fTest.shape[0]
        utest_Pred = h * G[:Idx+1] @ fTest
        url2 = relative_err(utest_Pred, uTest[:Idx+1])
        url2k = relative_err(utest_Pred[-1], uTest[:Idx+1][-1])

        if url2k > 1e-3:
            abnormals.append(Idx)

        if args.mode == 'sub':
            if Idx > 100:
                break 
        # np.save(upred_outpath, utest_Pred)
        # np.save(Gpred_outpath, G)

        print('Idx : {:} - url2k : {:.4e} - url2 : {:.4e} - Grl2 : {:.4e}'.format(Idx, url2k, url2, Grl2))  
    end_time = time.time()
    print(f"Single process training time: {end_time - start_time:.4f} seconds")

    print(abnormals)
    Gpred = np.concatenate(G)

    # save outputs
    exp_nm = 'ogapw-{:}-{:}-{:}-{:}-{:}-{:}-relu'.format(args.nNeuron, args.nTrain, args.nr, args.res, args.param, args.sigma)
    print(exp_nm)
    if args.mode == 'sub':
        exp_root = './results_sub'
    else:
        exp_root = './results'

    log_outpath, upred_outpath, model_outpath, Gpred_outpath = init_records(
            exp_root, args.task, exp_nm)

    # np.save(log_outpath, model.log)
    np.save(upred_outpath, utest_Pred)
    np.save(Gpred_outpath, Gpred)
    # np.save(model_outpath, model.model_weights)