import os 
import scipy 
import numpy as np
from einops import rearrange 
from utils import ReLU, RelativeErr, init_records
from utils import PoissonKernelDisk
from scipy.spatial.distance import cdist
from functools import partial
import warnings
import argparse
from tqdm import tqdm
warnings.filterwarnings("error")

class KernelOGA2D:
    def __init__(
            self, nNeuron, act, 
            x, y, 
            fs, us, fTest, uTest, 
            area, Gref=None, 
            nParam=40001):
        
        '''
        x  : nPts x 1 
        fs : nPts x nSample 
        us : nPts x nSample 
        Gref : nPts x nPts
        '''

        self.nNeuron = nNeuron 
        self.inpDim = x.shape[1] + y.shape[1]
        self.nxPts = x.shape[0]
        self.nyPts = y.shape[0]
        self.act = act 
        print("nxPts x nyPts : ", self.nxPts, self.nyPts)

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
        self.Gref = Gref 
        self.fs = fs 
        self.fTest = fTest 

        # self.us = us 
        # self.uTest = uTest
        self.us = self.h * Gref @ fs        
        self.uTest = self.h * Gref @ fTest

        # learned kernel
        self.Gk = np.zeros((self.nxPts, self.nyPts))
        self.gbasis = []
        self.uhbasis = []

        # neural network
        self.act = act
        self.Alpha = np.zeros((nNeuron, 1))
        self.WB = np.zeros((nNeuron, self.inpDim + 1))
        self.X = np.c_[xs, ys, np.ones((xs.shape[0],1))]

        # wb search 
        self.nParam = nParam
        print("nParam : ", self.nParam)
        self.brute_forward()

        # log 
        self.Glog = []
        self.ulog = []

    def brute_forward(self):
        self.WBs = (np.random.rand(self.nParam, self.inpDim+1) - 0.5)*2
        self.WBs[:,-1] = self.WBs[:,-1] * 2
        self.gs = self.act(self.WBs @ self.X.T).reshape(self.nParam, self.nxPts, self.nyPts)
        self.uhss =  self.h * self.gs @ self.fs
        print('basis calculation finish')
 
    def brute_search(self):
        # init search
        uts = self.h * self.Gk @ self.fs
        rG = self.h * ((self.us - uts) * self.uhss).sum(axis=(1,2))
        E = -0.5 * rG ** 2 
        idx = np.argmin(E)

        wbk = self.WBs[idx]
        gk = self.gs[idx]
        uhk = self.uhss[idx]

        return wbk, gk, uhk

    def projection(self):
        uhsub = np.array(self.uhbasis)
        A = self.h * np.einsum('kns,pns->kp', uhsub, uhsub)
        b = self.h * np.einsum('kns,ns->k', uhsub, self.us)
        alpha_k = scipy.linalg.solve(A, b)
        return alpha_k

    def optimize(self):
        for k in range(self.nNeuron):
            wbk, gk, uhk = self.brute_search()
            self.gbasis.append(gk)
            self.uhbasis.append(uhk)            
            try:
                alpha_k = self.projection()
            except:
                print("LinAlgWarning, we should stop adding neurons")
                break 
            
            # update params 
            self.WB[k] = wbk
            self.Alpha[:k+1] = alpha_k.reshape(-1,1)

            # update Gk
            self.Gk = (self.Alpha.T @ self.act(self.WB @ self.X.T)).reshape(self.nxPts, self.nyPts)

            self.uPred = self.h * self.Gk @ self.fTest
            url2 = RelativeErr(self.uPred.T, self.uTest.T)
            self.ulog.append(url2)

            if self.Gref is not None:
                Grl2 = RelativeErr(self.Gk.reshape(1,-1), self.Gref.reshape(1,-1))
                self.Glog.append(Grl2)            
            else:
                Grl2 = '-'
            
            if k % 1 == 0:
                print('{:}th (u) : {:.4e} (G) : {:.4e}'.format(k, url2, Grl2))

        
        self.model_weights = {"WB":self.WB, "Alpha":self.Alpha}
        self.log = {"Grl2" : np.array(self.Glog), "url2" : np.array(self.ulog)}
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGA for 2D kernel estimation')
    parser.add_argument('--task', type=str, default='helmholtz2D',
                        help='dataset name. (poisson2D, helmholtz2D)')
    parser.add_argument('--nNeuron', type=int, default=257,
                        help='maximum number of neurons')
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

    # meshy_path = os.path.join(data_root, 'mesh2D_{:}.mat'.format(args.res))
    # fs_path = os.path.join(data_root, 'dat2D_{:}.mat'.format(args.res))
    # us_path = os.path.join(data_root, f'{args.task}_{args.res}.mat')

    xs = scipy.io.loadmat(mesh_path)['X']
    ys = scipy.io.loadmat(meshy_path)['Y']
    fs = scipy.io.loadmat(fs_path)['Fy']
    us = scipy.io.loadmat(us_path)['U']

    # ys = scipy.io.loadmat(meshy_path)['X']
    # fs = scipy.io.loadmat(fs_path)['F']
    # us = scipy.io.loadmat(us_path)['U']

    nTrain = args.nTrain
    nTest = args.nTest

    fTrain = fs[:,:nTrain]
    fTest = fs[:,-nTest:]
    uTrain = us[:,:nTrain]
    uTest = us[:,-nTest:]
    
    if args.task == 'poisson2D':
        Gref = PoissonKernelDisk(xs, ys)
    else:
        Gref = None

    # network prepare
    nSample = nTrain
    nNeuron = args.nNeuron
    act = partial(ReLU, n=1)
    area = np.pi
    oga2d = KernelOGA2D(nNeuron, act, xs, ys, fTrain, uTrain, fTest, uTest, area, Gref)
    # model train
    oga2d.optimize()

    # save outputs
    log_outpath, upred_outpath, model_outpath, Gpred_outpath = init_records('./results', args.task, 'oga-{:}-{:}-{:}-relu'.format(args.nNeuron, args.nTrain, args.res))
    np.save(log_outpath, oga2d.log)
    np.save(upred_outpath, oga2d.uPred)
    np.save(Gpred_outpath, oga2d.Gk)
    np.save(model_outpath, oga2d.model_weights)