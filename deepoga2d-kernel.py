import os 
import scipy 
import torch 
import numpy as np
from einops import rearrange 
from utils import ReLU, Gauss
from utils import RelativeErr, init_records
from utils import PoissonKernelDisk
from utils import rational_mlp
from scipy.spatial.distance import cdist
from functools import partial
import warnings
import argparse
from tqdm import tqdm
warnings.filterwarnings("error")

class KernelDeepOGA2D:
    def __init__(
            self, 
            nBasis, 
            nLayer, nHidden,
            x, y, 
            fs, us, 
            fTest, uTest, 
            area,
            device, 
            Gref=None):
        
        '''
        x  : nPts x 1 
        fs : nPts x nSample 
        us : nPts x nSample 
        Gref : nPts x nPts
        '''

        # mesh 
        self.nxPts = x.shape[0]
        self.nyPts = y.shape[0]
        self.act = act 
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
        print("nxPts x nyPts : ", self.nxPts, self.nyPts)

        # dataset 
        self.Gref = Gref 
        self.fs = fs 
        self.fTest = fTest 

        # self.us = us 
        # self.uTest = uTest
        self.us = self.h * Gref @ fs        
        self.uTest = self.h * Gref @ fTest

        # deep neural basis
        self.device = device
        self.nBasis = nBasis 
        self.inpDim = x.shape[1] + y.shape[1]
        self.nLayer = nLayer 
        self.nHidden = nHidden
        self.Alpha = np.zeros((nBasis, 1))
        self.Gk = np.zeros((self.nxPts, self.nyPts))
        self.gbasis = []
        self.uhbasis = []

        # log 
        self.Glog = []
        self.ulog = []

    def basis_search(self):
        # init search


        mlp_basis = rational_mlp(self.inpDim, self.nLayer, self.nHidden)

        uts = self.h * self.Gk @ self.fs
        rG = self.h * ((self.us - uts) * self.uhss).sum(axis=(1,2))
        E = -0.5 * rG ** 2 
        idx = np.argmin(E)

        gk = self.gs[idx]
        uhk = self.uhss[idx]

        return gk, uhk

    def projection(self):
        uhsub = np.array(self.uhbasis)
        A = self.h * np.einsum('kns,pns->kp', uhsub, uhsub)
        b = self.h * np.einsum('kns,ns->k', uhsub, self.us)
        alpha_k = scipy.linalg.solve(A, b)
        return alpha_k

    def optimize(self):
        for k in range(1, self.nNeuron):
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
            self.Gk = np.einsum("nc, nwh -> cwh", self.Alpha[:k+1], np.array(self.gbasis))[0]

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

    # load pretrained gl model 
    gl_model = gl_pretrain_model(4, 5, 50, './results/poisson2D/pre-gl-rational/gl_state_dict.pt')

    # network prepare
    nSample = nTrain
    nNeuron = args.nNeuron
    act = np.tanh # partial(Gauss, c=10) # partial(ReLU, n=1)
    area = np.pi
    oga2d = KernelOGA2D(nNeuron, act, xs, ys, fTrain, uTrain, fTest, uTest, area, Gref)

    # calculate G0 
    oga2d.calc_g0(gl_model)

    # model train
    oga2d.optimize()

    # save outputs
    log_outpath, upred_outpath, model_outpath, Gpred_outpath = init_records('./results', args.task, 'oga+-{:}-{:}-{:}-relu'.format(args.nNeuron, args.nTrain, args.res))
    np.save(log_outpath, oga2d.log)
    np.save(upred_outpath, oga2d.uPred)
    np.save(Gpred_outpath, oga2d.Gk)
    np.save(model_outpath, oga2d.model_weights)