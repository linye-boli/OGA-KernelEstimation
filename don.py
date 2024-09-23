import os
import argparse
import numpy as np
import torch
import deepxde as dde 
from utils import init_records
from utils import relative_err
import scipy 

class DONFitter:
    def __init__(self, xs, fs, us, fTest, uTest):

        '''
        xs  : nPts x (nx + ny)
        fs : nPts x nSample 
        us : nPts x nSample 
        '''

        xs = xs.astype(np.float32)
        fs = fs.astype(np.float32)
        us = us.astype(np.float32)
        fTest = fTest.astype(np.float32)
        uTest = uTest.astype(np.float32)
        m, dim_x = xs.shape

        self.X_train = (fs.T, xs)
        self.y_train = us.T
        self.X_test = (fTest.T, xs)
        self.y_test = uTest.T        

        data = dde.data.TripleCartesianProd(
            X_train=self.X_train, y_train=self.y_train, 
            X_test=self.X_test, y_test=self.y_test)
        
        if dim_x == 1:
            branch = [m, 256, 128, 128, 128]
        elif dim_x == 2:
            branch = [m, 512, 256, 128, 128]
        elif dim_x == 3:
            branch = [m, 1024, 512, 256, 128]


        net = dde.nn.DeepONetCartesianProd(
            branch,
            [dim_x, 128, 128, 128, 128],
            "relu",
            "Glorot normal",)
        
        self.model = dde.Model(data, net)

    def optimize_adam(self, niter):
        self.model.compile("adam", lr=0.001, loss="mean l2 relative error", metrics=["mean l2 relative error"])
        losshistory, train_state = self.model.train(iterations=niter, batch_size=64, display_every=1000)
        self.log = {"utest_rl2" : losshistory.loss_test, "utrain_rl2" : losshistory.loss_train}

        upred = self.model.predict(self.X_test)
        utest_rl2 = relative_err(upred, self.y_test)
        print("test rl2 : {:.4e}".format(utest_rl2))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGA for kernel estimation')
    parser.add_argument('--task', type=str, default='poisson1D',
                        help='dataset name. (poisson1D, helmholtz1D, airy1D, poisson2D)')
    parser.add_argument('--nIter', type=int, default=2000, 
                        help='maximum number of neurons')
    parser.add_argument('--nTrain', type=int, default=200, 
                        help='number of training samples')
    parser.add_argument('--nTest', type=int, default=200, 
                        help='number of test samples')
    args = parser.parse_args()
        
    if args.task == 'poisson1D':
        from utils import load_poisson1d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_poisson1d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
        xs = scipy.io.loadmat('./data/mesh.mat')['X'][::4]
    elif args.task == 'helmholtz1D':
        from utils import load_helmholtz1d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmholtz1d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
        xs = scipy.io.loadmat('./data/mesh.mat')['X'][::4]
    elif args.task == 'poisson2D':
        from utils import load_poisson2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_poisson2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
        xs = scipy.io.loadmat('./data/mesh2D_disk.mat')['X']
    elif args.task == 'helmholtz2D':
        from utils import load_helmholtz2d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmholtz2d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
        xs = scipy.io.loadmat('./data/mesh2D_disk.mat')['X']
    elif args.task == 'poisson2Dhdomain':
        from utils import load_poisson2dhdomain_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_poisson2dhdomain_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
        xs = scipy.io.loadmat('./data/mesh2D_h.mat')['X']
    elif args.task == 'helmholtz2Dhdomain':
        from utils import load_helmholtz2dhdomain_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_helmholtz2dhdomain_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
        xs = scipy.io.loadmat('./data/mesh2D_h.mat')['X']
    elif args.task == 'log3D':
        from utils import load_log3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_log3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
        xs = scipy.io.loadmat('./data/mesh3D_box_17.mat')['X']
    elif args.task == 'logsin3D':
        from utils import load_logsin3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_logsin3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
        xs = scipy.io.loadmat('./data/mesh3D_box_17.mat')['X']
    elif args.task == 'cos3D':
        from utils import load_cos3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_cos3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
        xs = scipy.io.loadmat('./data/mesh3D_box_17.mat')['X']
    elif args.task == 'logcos3D':
        from utils import load_logcos3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_logcos3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest)
        xs = scipy.io.loadmat('./data/mesh3D_box_17.mat')['X']

    model = DONFitter(
        xs = xs,
        fs=fTrain, us=uTrain, 
        fTest=fTest, uTest=uTest)

    # model train
    model.optimize_adam(args.nIter)

    # # save outputs
    log_outpath, upred_outpath, model_outpath, Gpred_outpath = init_records('./results', args.task, 'don-{:}-{:}'.format(args.nIter, args.nTrain))

    np.save(log_outpath, model.log)
    # np.save(upred_outpath, model.utest_Pred)