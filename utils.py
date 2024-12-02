import numpy as np 
import scipy 
from scipy.special import expit
import os 

def rotation(ys, angle_degrees):
    x, y = ys[:,0], ys[:,1]
    angle_radians = np.radians(angle_degrees)
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    
    # Original coordinates
    original_point = np.array([x, y])
    
    # Rotated coordinates
    rotated_point = rotation_matrix.dot(original_point).T
    
    return rotated_point

def Gauss(x, c):
    return np.exp(-x**2/c)

def ReLU(x, n):
    return np.maximum(0, x) ** n 

def Sigmoid(x):
    return expit(x)

def PoissonKernelDisk(xs, ys):
    X = np.c_[xs, ys]
    a = (X[:,[0]] - X[:,[2]])**2
    b = (X[:,[1]] - X[:,[3]])**2
    c = (X[:,[0]]*X[:,[3]] - X[:,[2]]*X[:,[1]])**2
    d = (X[:,[0]]*X[:,[2]] + X[:,[3]]*X[:,[1]]-1)**2
    Gref = (0.25/np.pi) * (np.log(a + b) - np.log(c + d))

    return Gref

def PoissonKernelFree(xs, ys):
    X = np.c_[xs, ys]
    a = (X[:,[0]] - X[:,[2]])**2
    b = (X[:,[1]] - X[:,[3]])**2
    Gref = np.log(a + b) / (4*np.pi)
    # Gref = np.log(c + d)
    return Gref

def HelmholtzKernelReal(xs, ys, k=2):
    X = np.c_[xs, ys]
    a = (X[:,[0]] - X[:,[2]])**2
    b = (X[:,[1]] - X[:,[3]])**2
    r = (a+b)**0.5 
    Gref = -np.cos(np.pi*k*r)/(4*np.pi*r)
    Gref = np.clip(Gref, a_min=-6, a_max=2)
    # Gref = np.log(c + d)
    return Gref

def HelmholtzKernelImg(xs, ys, k=2):
    X = np.c_[xs, ys]
    a = (X[:,[0]] - X[:,[2]])**2
    b = (X[:,[1]] - X[:,[3]])**2
    r = (a+b)**0.5 
    Gref = np.sin(np.pi*k*r)/(4*np.pi*r)
    # Gref = np.log(c + d)
    return Gref

def CosineKernel2D(xs, ys, n=1):
    X = np.c_[xs, ys]
    a = (X[:,[0]] - X[:,[2]])**2
    b = (X[:,[1]] - X[:,[3]])**2
    r = (a + b)**0.5
    Gref = np.cos(np.pi*n*r)
    return Gref

def SineKernel2D(xs, ys, n=0.5):
    X = np.c_[xs, ys]
    a = X[:,[0]]
    c = X[:,[2]]
    b = X[:,[1]]
    d = X[:,[3]]

    Gref = np.sin(n*np.pi*a) * np.sin(n*np.pi*b) * np.sin(n*np.pi*c) * np.sin(n*np.pi*d)
    return Gref

def GaussKernel2D(xs, ys, C=0.1):
    X = np.c_[xs, ys]
    a = (X[:,[0]] - X[:,[2]])**2
    b = (X[:,[1]] - X[:,[3]])**2
    r = (a+b)**0.5
    
    Gref = np.exp(-C*r)
    return Gref

def LogKernel2D(xs, ys):
    X = np.c_[xs, ys]
    a = (X[:,[0]] - X[:,[2]])**2
    b = (X[:,[1]] - X[:,[3]])**2
    r = (a + b)**0.5
    r[r== 0] = 0.03125

    Gref = np.log(r)
    return Gref

def CosineKernel3D(xs, ys, n=2):
    X = np.c_[xs, ys]
    a = (X[:,[0]] - X[:,[3]])**2
    b = (X[:,[1]] - X[:,[4]])**2
    c = (X[:,[2]] - X[:,[5]])**2
    r = (a + b + c)**0.5
    Gref = np.cos(n*np.pi*r)
    print(f"estimate cos(r) kernel, k={n}")
    return Gref

def LogKernel3D(xs, ys):
    X = np.c_[xs, ys]
    a = (X[:,[0]] - X[:,[3]])**2
    b = (X[:,[1]] - X[:,[4]])**2
    c = (X[:,[2]] - X[:,[5]])**2
    r = (a + b + c)**0.5

    r[r== 0] = 0.03125

    Gref = np.log(r)
    print("estimate log(r) kernel")
    return Gref

def LogSinKernel3D(xs, ys, n=2):
    X = np.c_[xs, ys]
    a = (X[:,[0]] - X[:,[3]])**2
    b = (X[:,[1]] - X[:,[4]])**2
    c = (X[:,[2]] - X[:,[5]])**2
    r = (a + b + c)**0.5
    r[r== 0] = 0.03125

    Gref = np.log(r) * np.sin(n*np.pi*r)
    print(f"estimate log(r)*sin(kr) kernel, k={n}")
    return Gref

def LogCosKernel3D(xs, ys, k=2):
    X = np.c_[xs, ys]
    a = (X[:,[0]] - X[:,[3]])**2
    b = (X[:,[1]] - X[:,[4]])**2
    c = (X[:,[2]] - X[:,[5]])**2
    r = (a + b + c)**0.5
    r[r== 0] = 0.03125

    Gref = np.log(r) * np.cos(k*np.pi*r)
    print(f"estimate log(r)*cos(kr) kernel, k={k}")
    return Gref

def InvKernel3D(xs, ys):
    X = np.c_[xs, ys]
    a = (X[:,[0]] - X[:,[3]])**2
    b = (X[:,[1]] - X[:,[4]])**2
    c = (X[:,[2]] - X[:,[5]])**2
    r = (a + b + c)**0.5

    r[r== 0] = 0.03125
    
    Gref = -1/(4*np.pi*r)
    return Gref


# modify from https://github.com/hsharsh/EmpiricalGreensFunctions.git
def PoissonKernel1D(domain):
    x, s = np.meshgrid(domain,domain)
    G = np.empty(np.shape(x))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            xx, ss = x[i,j], s[i,j]
            if xx <= ss:
                G[i,j] = xx * (1-ss)
            else:
                G[i,j] = ss * (1-xx)
    return G 

def ADKernel1D(domain):
    x, s = np.meshgrid(domain,domain)
    G = np.empty(np.shape(x))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            xx, ss = x[i,j], s[i,j]
            if xx <= ss:
                G[i,j] = 4*xx*(ss-1) * np.exp(-2*(xx-ss))
            else:
                G[i,j] = ss * (xx-1)
    return G 

def SineKernel1D(domain):
    x, s = np.meshgrid(domain,domain)
    G = np.empty(np.shape(x))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            xx, ss = x[i,j], s[i,j]
            G[i,j] = np.sin(np.pi * xx) * np.sin(np.pi * ss)
    return G 

def GaborKernel1D(domain, n=4):
    x, s = np.meshgrid(domain,domain)
    G = np.empty(np.shape(x))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            xx, ss = x[i,j], s[i,j]
            G[i,j] = np.exp(-((xx - 0.5) ** 2 + (ss - 0.5) ** 2) / (2 * 0.15 ** 2) ) * np.cos(n*np.pi * xx)
    return G 

def HelmholtzKernel1D(domain, K=15):
    x, s = np.meshgrid(domain,domain)
    G = np.empty(np.shape(x))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            xx, ss = x[i,j], s[i,j]
            if xx <= ss:
                G[i,j] = np.sin(K*xx) * np.sin(K*(ss-1))/(K*np.sin(K))
            else:
                G[i,j] = np.sin(K*ss) * np.sin(K*(xx-1))/(K*np.sin(K))
    return G

def relative_err(u_est, u_ref):
    # u_est, u_ref : nSample x nPts
    # import pdb 
    # pdb.set_trace()

    u_est = u_est.reshape(-1,1)
    u_ref = u_ref.reshape(-1,1)

    if isinstance(u_est, np.ndarray):
        dif_norm = np.linalg.norm(u_est - u_ref, ord=2)
        ref_norm = np.linalg.norm(u_ref, 2)
    else:
        dif_norm = torch.linalg.norm(u_est - u_ref, ord=2)
        ref_norm = torch.linalg.norm(u_ref, 2)

    return ((dif_norm) / (ref_norm)).mean() 
    # return dif_norm.mean()

def init_records(log_root, task_nm, exp_nm):
    exp_root = os.path.join(log_root, task_nm, exp_nm)
    os.makedirs(exp_root, exist_ok=True)

    log_outpath = os.path.join(exp_root, 'log.npy')
    upred_outpath = os.path.join(exp_root, 'upred.npy')
    model_outpath = os.path.join(exp_root, 'model.npy')
    Gpred_outpath = os.path.join(exp_root, 'Goga.npy')
    
    return log_outpath, upred_outpath, model_outpath, Gpred_outpath

import torch 
import torch.nn as nn

def save_pytorch_model(log_root, task_nm, exp_nm, model):
    exp_root = os.path.join(log_root, task_nm, exp_nm)
    os.makedirs(exp_root, exist_ok=True)
    model_outpath = os.path.join(exp_root, 'gl_state_dict.pt')
    print(model_outpath)
    torch.save(model.state_dict(), model_outpath)

def load_sin1d_fitting_dataset(nTrain=1024, nx=100):
    def f1d(x):
        return np.sin(2*np.pi*x)

    # generate training data
    # np.random.seed(10)
    X = np.random.rand(nTrain, 1)
    y = f1d(X)
    
    Xtest = np.linspace(0,1,nx).reshape(-1,1)
    ytest = f1d(Xtest)

    return X, y, Xtest, ytest

def load_gauss1d_fitting_dataset(nTrain=10000, nx=101, L=1000):
    def f1d(x, L):
        return np.exp(-L * x**2)

    # generate training data
    np.random.seed(0)
    X = np.random.rand(nTrain, 1) * 2 - 1
    y = f1d(X, L)

    Xtest = np.linspace(-1,1,nx).reshape(-1,1)
    ytest = f1d(Xtest, L)

    return X, y, Xtest, ytest

def load_gabor2d_fitting_dataset(nTrain=2500, nx=101):
    def f2d(X):
        x = X[:,[0]]
        y = X[:,[1]]
        sigma = 0.15 
        m = 8
        return np.exp(-((x-0.5)**2 + (y-0.5)**2)/(2*sigma**2)) * np.cos(2*np.pi * m * x)

    # generate training data
    np.random.seed(0)
    X = np.random.rand(nTrain, 2)
    y = f2d(X)

    x = np.linspace(0,1,nx)
    xx, yy = np.meshgrid(x, x)
    Xtest = np.c_[xx.reshape(-1), yy.reshape(-1)]
    ytest = f2d(Xtest)

    return X, y, Xtest, ytest

def load_sin4d_fitting_dataset(nTrain=500000, nTest=500000):
    def f4d(X):
        x1 = X[:,[0]]
        x2 = X[:,[1]]
        y1 = X[:,[2]]
        y2 = X[:,[3]]
        return np.sin(np.pi*x1) * np.sin(np.pi*y1) * np.sin(np.pi*x2) * np.sin(np.pi*y2)

    # generate training data
    X = np.random.rand(nTrain,4)
    y = f4d(X)

    # generate test data
    Xtest = np.random.rand(nTest,4)
    ytest = f4d(Xtest)

    return X, y, Xtest, ytest

def load_rbf1d_fitting_dataset(nTrain=50000, nx=101, L=1000):
    def f2d(X, L):
        x = X[:,[0]]
        y = X[:,[1]]
        return np.exp(-L * (x-y)**2)

    # generate training data
    np.random.seed(0)
    X = np.random.rand(nTrain, 2)
    y = f2d(X, L)

    x = np.linspace(0,1,nx)
    xx, yy = np.meshgrid(x, x)
    Xtest = np.c_[xx.reshape(-1), yy.reshape(-1)]
    ytest = f2d(Xtest, L)

    return X, y, Xtest, ytest

def load_rbf2d_fitting_dataset(nTrain=500000, nTest=500000, L=100):
    def f4d(X, L):
        x1 = X[:,[0]]
        x2 = X[:,[1]]
        y1 = X[:,[2]]
        y2 = X[:,[3]]

        return np.exp(-L * ((x1-y1)**2 + (x2-y2)**2))

    # generate training data
    X = np.random.rand(nTrain,4)
    y = f4d(X, L) 

    # generate test data
    Xtest = np.random.rand(nTest,4)
    ytest = f4d(Xtest, L)

    return X, y, Xtest, ytest

def load_runge1d_fitting_dataset(nTrain=10000, nx=100, L=100):
    def f1d(x, C):
        return 1/(C * x**2 + 1)

    # generate training data
    np.random.seed(0)
    X = np.random.rand(nTrain, 1) * 2 - 1
    y = f1d(X, L)

    Xtest = np.linspace(-1,1,nx).reshape(-1,1)
    ytest = f1d(Xtest, L)

    return X, y, Xtest, ytest

def load_oscillatory1d_fitting_dataset(nTrain=10000, nx=100):
    # def f1d(x):
    #     return np.cos(6*np.pi*x)**2 + np.sin(10*np.pi * x**2)
    
    def f1d(x):
        return np.sin(50*np.pi*x)
        # return np.cos(6*np.pi*x)**2 + np.sin(10*np.pi * x**2)

    # generate training data
    np.random.seed(0)
    X = np.random.rand(nTrain, 1) * 2 - 1
    y = f1d(X)

    Xtest = np.linspace(-1,1,nx).reshape(-1,1)
    ytest = f1d(Xtest)

    return X, y, Xtest, ytest

def load_arctan1d_fitting_dataset(nTrain=10000, nx=100):
    def f1d(x):
        return np.arctan(100*x+20)

    # generate training data
    np.random.seed(0)
    X = np.random.rand(nTrain, 1) * 2 - 1
    y = f1d(X)

    Xtest = np.linspace(-1,1,nx).reshape(-1,1)
    ytest = f1d(Xtest)

    return X, y, Xtest, ytest

def load_localoscillatory1d_fitting_dataset(nTrain=10000, nx=100):
    def f1d(x):
        mask = np.abs(x + 0.2) < 0.02
        return mask * np.sin(50*np.pi*x)

    # generate training data
    np.random.seed(0)
    X = np.random.rand(nTrain, 1) * 2 - 1
    y = f1d(X)

    Xtest = np.linspace(-1,1,nx).reshape(-1,1)
    ytest = f1d(Xtest)

    return X, y, Xtest, ytest

def load_poisson1d_fitting_dataset(nTrain=5000, nx=101):
    def f2d(X):
        return 0.5 * (X[:,[0]] + X[:,[1]] - np.abs(X[:,[0]]-X[:,[1]])) - X[:,[0]]*X[:,[1]]

    # generate training data
    np.random.seed(0)
    X = np.random.rand(nTrain, 2)
    y = f2d(X)

    x = np.linspace(0,1,nx)
    xx, yy = np.meshgrid(x, x)
    Xtest = np.c_[xx.reshape(-1), yy.reshape(-1)]
    ytest = f2d(Xtest)

    return X, y, Xtest, ytest

def load_poisson2d_fitting_dataset(nTrain=50000, nTest=5000):
    def f4d(X):
        a = (X[:,[0]] - X[:,[2]])**2
        b = (X[:,[1]] - X[:,[3]])**2
        c = (X[:,[0]]*X[:,[3]] - X[:,[2]]*X[:,[1]])**2
        d = (X[:,[0]]*X[:,[2]] + X[:,[3]]*X[:,[1]]-1)**2
        
        # return (0.25/np.pi) * np.log((a + b))
        return (0.25/np.pi) * np.log((a + b)/(c + d))

    # generate training data
    R_x = np.random.rand(nTrain)
    R_y = np.random.rand(nTrain)
    Theta_x = (np.random.rand(nTrain)*2 - 1) * np.pi
    Theta_y = (np.random.rand(nTrain)*2 - 1) * np.pi    
    X = np.c_[R_x * np.cos(Theta_x), R_x * np.sin(Theta_x), R_y * np.cos(Theta_y), R_y * np.sin(Theta_y)]
    y = f4d(X)
    # X[:,0] = (X[:,0] + 1)/2
    # X[:,1] = (X[:,1] + 1)/2
    # X[:,2] = (X[:,2] + 1)/2
    # X[:,3] = (X[:,3] + 1)/2

    # generate test data
    R_x = np.random.rand(nTest)
    R_y = np.random.rand(nTest)
    Theta_x = (np.random.rand(nTest)*2 - 1) * np.pi
    Theta_y = (np.random.rand(nTest)*2 - 1) * np.pi    
    
    Xtest = np.c_[R_x * np.cos(Theta_x), R_x * np.sin(Theta_x), R_y * np.cos(Theta_y), R_y * np.sin(Theta_y)]
    ytest = f4d(Xtest)

    # Xtest[:,0] = (Xtest[:,0] + 1)/2
    # Xtest[:,1] = (Xtest[:,1] + 1)/2
    # Xtest[:,2] = (Xtest[:,2] + 1)/2
    # Xtest[:,3] = (Xtest[:,3] + 1)/2

    return X, y, Xtest, ytest

def load_log1d_fitting_dataset(nTrain=10000, nTest=5000):
    def f2d(X):
        return np.log(np.abs(X[:,[0]]-X[:,[1]]))

    # generate training data
    np.random.seed(0)
    X = np.random.rand(nTrain, 2)
    y = f2d(X)

    Xtest = np.random.rand(nTest, 2)
    ytest = f2d(Xtest)

    return X, y, Xtest, ytest

def load_log2d_fitting_dataset(nTrain=50000, nTest=5000):
    def f4d(X):
        a = (X[:,[0]] - X[:,[2]])**2
        b = (X[:,[1]] - X[:,[3]])**2
        
        return np.log((a + b)**0.5)

    # generate training data
    np.random.seed(0)
    X = np.random.rand(nTrain, 4)
    y = f4d(X)

    # generate test data
    Xtest = np.random.rand(nTest, 4)
    ytest = f4d(Xtest)

    return X, y, Xtest, ytest

def load_poisson2d_fitting_dataset(nTrain=5000, nTest=5000):
    def f4d(X):
        a = (X[:,[0]] - X[:,[2]])**2
        b = (X[:,[1]] - X[:,[3]])**2
        c = (X[:,[0]]*X[:,[3]] - X[:,[2]]*X[:,[1]])**2
        d = (X[:,[0]]*X[:,[2]] + X[:,[3]]*X[:,[1]]-1)**2
        
        # return (0.25/np.pi) * np.log((a + b))
        return (0.25/np.pi) * np.log((a + b)/(c + d))

    # generate training data
    R_x = np.random.rand(nTrain)
    R_y = np.random.rand(nTrain)
    Theta_x = (np.random.rand(nTrain)*2 - 1) * np.pi
    Theta_y = (np.random.rand(nTrain)*2 - 1) * np.pi    
    X = np.c_[R_x * np.cos(Theta_x), R_x * np.sin(Theta_x), R_y * np.cos(Theta_y), R_y * np.sin(Theta_y)]
    y = f4d(X)

    # X[:,0] = (X[:,0] + 1)/2
    # X[:,1] = (X[:,1] + 1)/2
    # X[:,2] = (X[:,2] + 1)/2
    # X[:,3] = (X[:,3] + 1)/2

    # generate test data
    R_x = np.random.rand(nTest)
    R_y = np.random.rand(nTest)
    Theta_x = (np.random.rand(nTest)*2 - 1) * np.pi
    Theta_y = (np.random.rand(nTest)*2 - 1) * np.pi    
    
    Xtest = np.c_[R_x * np.cos(Theta_x), R_x * np.sin(Theta_x), R_y * np.cos(Theta_y), R_y * np.sin(Theta_y)]
    ytest = f4d(Xtest)

    # Xtest[:,0] = (Xtest[:,0] + 1)/2
    # Xtest[:,1] = (Xtest[:,1] + 1)/2
    # Xtest[:,2] = (Xtest[:,2] + 1)/2
    # Xtest[:,3] = (Xtest[:,3] + 1)/2

    return X, y, Xtest, ytest

# ---------------------------------------------------------------------
# kernel fitting dataset 
# ---------------------------------------------------------------------

def load_poisson1d_kernel_dataset(
        data_root, nTrain, nTest):
    
    mesh_path = os.path.join(data_root, 'mesh.mat')
    fs_path = os.path.join(data_root, 'dat1D.mat')
    us_path = os.path.join(data_root, 'poisson1D.mat')

    xs = scipy.io.loadmat(mesh_path)['X'][::4]
    xs = (xs + 1)/2 
    nxPts = xs.shape[0]
    nyPts = nxPts
    Gref = PoissonKernel1D(xs)

    fs = scipy.io.loadmat(fs_path)['F'][::4]
    us = scipy.io.loadmat(us_path)['U'][::4]

    if nTrain + nTest > fs.shape[1]:
         raise AssertionError("train set and test set has overlap")

    fTrain = fs[:,:nTrain]
    fTest = fs[:,-nTest:]
    uTrain = us[:,:nTrain]
    uTest = us[:,-nTest:]


    xx, yy = np.meshgrid(xs,xs)
    xs, ys = xx.reshape(-1), yy.reshape(-1)

    X = np.c_[xs, ys]

    Gref = Gref.reshape(nxPts, nyPts)
    h = 1 / fTrain.shape[0]
    uTrain = h * Gref @ fTrain 
    uTest = h * Gref @ fTest

    return fTrain, fTest, uTrain, uTest, X, Gref

def load_helmholtz1d_kernel_dataset(
        data_root, nTrain, nTest):
    
    mesh_path = os.path.join(data_root, 'mesh.mat')
    fs_path = os.path.join(data_root, 'dat1D.mat')
    us_path = os.path.join(data_root, 'helmholtz1D.mat')

    xs = scipy.io.loadmat(mesh_path)['X'][::4]
    xs = (xs + 1)/2 
    nxPts = xs.shape[0]
    nyPts = nxPts
    Gref = HelmholtzKernel1D(xs)

    fs = scipy.io.loadmat(fs_path)['F'][::4]
    us = scipy.io.loadmat(us_path)['U'][::4]

    if nTrain + nTest > fs.shape[1]:
         raise AssertionError("train set and test set has overlap")
    
    fTrain = fs[:,:nTrain]
    fTest = fs[:,-nTest:]
    uTrain = us[:,:nTrain]
    uTest = us[:,-nTest:]


    xx, yy = np.meshgrid(xs,xs)
    xs, ys = xx.reshape(-1), yy.reshape(-1)

    X = np.c_[xs, ys]

    Gref = Gref.reshape(nxPts, nyPts)
    h = 1 / fTrain.shape[0]
    uTrain = h * Gref @ fTrain 
    uTest = h * Gref @ fTest

    return fTrain, fTest, uTrain, uTest, X, Gref

def load_cos2d_kernel_dataset(
        data_root, nTrain, nTest, n, res=20, sigma='2e-01'):
    
    mesh_path = os.path.join(data_root, 'mesh2D_disk_{:}.mat'.format(res))
    meshy_path = os.path.join(data_root, 'mesh2D_disk_{:}.mat'.format(res))
    fs_path = os.path.join(data_root, 'dat2D_disk_{:}_{:}.mat'.format(res, sigma))

    print(f"load {mesh_path}")
    print(f"load {fs_path}")
    
    xs = scipy.io.loadmat(mesh_path)['X']
    ys = scipy.io.loadmat(meshy_path)['X']
    fs = scipy.io.loadmat(fs_path)['F']

    if nTrain + nTest > fs.shape[1]:
        raise AssertionError("train set and test set has overlap")

    fTrain = fs[:,:nTrain]
    fTest = fs[:,-nTest:]
    
    nxPts = xs.shape[0]
    nyPts = ys.shape[0]
    idx = np.arange(nxPts)
    idy = np.arange(nyPts)
    idxx, idyy = np.meshgrid(idx, idy)
    xxs = xs[idyy.reshape(-1)]
    yys = ys[idxx.reshape(-1)]
    X = np.c_[xxs, yys]
    Gref = CosineKernel2D(xxs, yys,n).reshape(nyPts, nxPts).T

    h = np.pi / fTrain.shape[0]
    uTrain = h * Gref @ fTrain 
    uTest = h * Gref @ fTest

    return fTrain, fTest, uTrain, uTest, X, Gref

def load_cos2dhdomain_kernel_dataset(
        data_root, nTrain, nTest, n, res=30, sigma='2e-01'):
    
    mesh_path = os.path.join(data_root, 'mesh2D_h_{:}.mat'.format(res))
    meshy_path = os.path.join(data_root, 'mesh2D_h_{:}.mat'.format(res))
    fs_path = os.path.join(data_root, 'dat2D_h_{:}_{:}.mat'.format(res, sigma))

    print(f"load {mesh_path}")
    print(f"load {fs_path}")

    xs = scipy.io.loadmat(mesh_path)['X']
    ys = scipy.io.loadmat(meshy_path)['X']
    fs = scipy.io.loadmat(fs_path)['F']

    if nTrain + nTest > fs.shape[1]:
        raise AssertionError("train set and test set has overlap")

    fTrain = fs[:,:nTrain]
    fTest = fs[:,-nTest:]

    nxPts = xs.shape[0]
    nyPts = ys.shape[0]
    idx = np.arange(nxPts)
    idy = np.arange(nyPts)
    idxx, idyy = np.meshgrid(idx, idy)
    xxs = xs[idyy.reshape(-1)]
    yys = ys[idxx.reshape(-1)]
    X = np.c_[xxs, yys]
    
    Gref = CosineKernel2D(xxs, yys,n).reshape(nyPts, nxPts).T
    
    h = np.pi / fTrain.shape[0]
    uTrain = h * Gref @ fTrain 
    uTest = h * Gref @ fTest

    return fTrain, fTest, uTrain, uTest, X, Gref

def load_poisson2d_kernel_dataset(
        data_root, nTrain, nTest, res=20, sigma='2e-01'):
    
    mesh_path = os.path.join(data_root, 'mesh2D_disk_{:}.mat'.format(res))
    meshy_path = os.path.join(data_root, 'mesh2D_disk_{:}.mat'.format(res))
    fs_path = os.path.join(data_root, 'dat2D_disk_{:}_{:}.mat'.format(res, sigma))
    us_path = os.path.join(data_root, 'poisson2D_disk_{:}_{:}.mat'.format(res, sigma))

    print(f"load {mesh_path}")
    print(f"load {fs_path}")
    print(f"load {us_path}")

    xs = scipy.io.loadmat(mesh_path)['X']
    ys = scipy.io.loadmat(meshy_path)['X']
    fs = scipy.io.loadmat(fs_path)['F']
    us = scipy.io.loadmat(us_path)['U']

    if nTrain + nTest > fs.shape[1]:
        raise AssertionError("train set and test set has overlap")

    fTrain = fs[:,:nTrain]
    fTest = fs[:,-nTest:]
    uTrain = us[:,:nTrain]
    uTest = us[:,-nTest:]
    
    nxPts = xs.shape[0]
    nyPts = ys.shape[0]
    idx = np.arange(nxPts)
    idy = np.arange(nyPts)
    idxx, idyy = np.meshgrid(idx, idy)
    xxs = xs[idyy.reshape(-1)]
    yys = ys[idxx.reshape(-1)]
    X = np.c_[xxs, yys]
    Gref = PoissonKernelDisk(xxs, yys).reshape(nxPts, nyPts)

    return fTrain, fTest, uTrain, uTest, X, Gref

def load_poisson2dhdomain_kernel_dataset(
        data_root, nTrain, nTest, res=30, sigma='2e-01'):
    
    mesh_path = os.path.join(data_root, 'mesh2D_h_{:}.mat'.format(res))
    meshy_path = os.path.join(data_root, 'mesh2D_h_{:}.mat'.format(res))
    fs_path = os.path.join(data_root, 'dat2D_h_{:}_{:}.mat'.format(res, sigma))
    us_path = os.path.join(data_root, 'poisson2D_h_{:}_{:}.mat'.format(res, sigma))

    print(f"load {mesh_path}")
    print(f"load {fs_path}")
    print(f"load {us_path}")

    xs = scipy.io.loadmat(mesh_path)['X']
    ys = scipy.io.loadmat(meshy_path)['X']
    fs = scipy.io.loadmat(fs_path)['F']
    us = scipy.io.loadmat(us_path)['U']

    if nTrain + nTest > fs.shape[1]:
        raise AssertionError("train set and test set has overlap")

    fTrain = fs[:,:nTrain]
    fTest = fs[:,-nTest:]
    uTrain = us[:,:nTrain]
    uTest = us[:,-nTest:]
    
    nxPts = xs.shape[0]
    nyPts = ys.shape[0]
    idx = np.arange(nxPts)
    idy = np.arange(nyPts)
    idxx, idyy = np.meshgrid(idx, idy)
    xxs = xs[idyy.reshape(-1)]
    yys = ys[idxx.reshape(-1)]
    X = np.c_[xxs, yys]
    Gref = None

    return fTrain, fTest, uTrain, uTest, X, Gref

def load_helmholtz2d_kernel_dataset(
        data_root, nTrain, nTest, res=20, sigma='2e-01'):
    
    mesh_path = os.path.join(data_root, 'mesh2D_disk_{:}.mat'.format(res))
    meshy_path = os.path.join(data_root, 'mesh2D_disk_{:}.mat'.format(res))
    fs_path = os.path.join(data_root, 'dat2D_disk_{:}_{:}.mat'.format(res, sigma))
    us_path = os.path.join(data_root, 'helmholtz2D_disk_{:}_{:}.mat'.format(res, sigma))

    print(f"load {mesh_path}")
    print(f"load {fs_path}")
    print(f"load {us_path}")

    xs = scipy.io.loadmat(mesh_path)['X']
    ys = scipy.io.loadmat(meshy_path)['X']
    fs = scipy.io.loadmat(fs_path)['F']
    us = scipy.io.loadmat(us_path)['U']

    fTrain = fs[:,:nTrain]
    fTest = fs[:,-nTest:]
    uTrain = us[:,:nTrain]
    uTest = us[:,-nTest:]
    
    nxPts = xs.shape[0]
    nyPts = ys.shape[0]
    idx = np.arange(nxPts)
    idy = np.arange(nyPts)
    idxx, idyy = np.meshgrid(idx, idy)
    xs = xs[idyy.reshape(-1)]
    ys = ys[idxx.reshape(-1)]
    X = np.c_[xs, ys]
    Gref = None 

    return fTrain, fTest, uTrain, uTest, X, Gref

def load_helmholtz2dhdomain_kernel_dataset(
        data_root, nTrain, nTest, res=30, sigma='2e-01'):
    
    mesh_path = os.path.join(data_root, 'mesh2D_h_{:}.mat'.format(res))
    meshy_path = os.path.join(data_root, 'mesh2D_h_{:}.mat'.format(res))
    fs_path = os.path.join(data_root, 'dat2D_h_{:}_{:}.mat'.format(res, sigma))
    us_path = os.path.join(data_root, 'helmholtz2D_h_{:}_{:}.mat'.format(res, sigma))

    print(f"load {mesh_path}")
    print(f"load {fs_path}")
    print(f"load {us_path}")

    xs = scipy.io.loadmat(mesh_path)['X']
    ys = scipy.io.loadmat(meshy_path)['X']
    fs = scipy.io.loadmat(fs_path)['F']
    us = scipy.io.loadmat(us_path)['U']

    fTrain = fs[:,:nTrain]
    fTest = fs[:,-nTest:]
    uTrain = us[:,:nTrain]
    uTest = us[:,-nTest:]
    
    nxPts = xs.shape[0]
    nyPts = ys.shape[0]
    idx = np.arange(nxPts)
    idy = np.arange(nyPts)
    idxx, idyy = np.meshgrid(idx, idy)
    xs = xs[idyy.reshape(-1)]
    ys = ys[idxx.reshape(-1)]
    X = np.c_[xs, ys]
    Gref = None 

    return fTrain, fTest, uTrain, uTest, X, Gref

def load_cos3d_kernel_dataset(
        data_root, nTrain, nTest, n=1, res=17, sigma='5e-01'):
    
    mesh_path = os.path.join(data_root, f'mesh3D_box_{res}.mat')
    meshy_path = os.path.join(data_root, f'mesh3D_box_{res}.mat')
    fs_path = os.path.join(data_root, f'dat3D_box_{res}_{sigma}.mat')

    xs = scipy.io.loadmat(mesh_path)['X']
    ys = scipy.io.loadmat(meshy_path)['X']
    fs = scipy.io.loadmat(fs_path)['F']

    fTrain = fs[:,:nTrain]
    fTest = fs[:,-nTest:]
    
    nxPts = xs.shape[0]
    nyPts = ys.shape[0]
    idx = np.arange(nxPts)
    idy = np.arange(nyPts)
    idxx, idyy = np.meshgrid(idx, idy)

    xs = xs[idyy.reshape(-1)]
    ys = ys[idxx.reshape(-1)]
    X = np.c_[xs, ys]
    Gref = CosineKernel3D(xs, ys, n).reshape(nxPts, nyPts)

    h = 1 / fTrain.shape[0]
    uTrain = h * Gref @ fTrain 
    uTest = h * Gref @ fTest

    return fTrain, fTest, uTrain, uTest, X, Gref

def load_logcos3d_kernel_dataset(
        data_root, nTrain, nTest, n=1, res=17, sigma='5e-01'):
    
    mesh_path = os.path.join(data_root, f'mesh3D_box_{res}.mat')
    meshy_path = os.path.join(data_root, f'mesh3D_box_{res}.mat')
    fs_path = os.path.join(data_root, f'dat3D_box_{res}_{sigma}.mat')

    xs = scipy.io.loadmat(mesh_path)['X']
    ys = scipy.io.loadmat(meshy_path)['X']
    fs = scipy.io.loadmat(fs_path)['F']

    fTrain = fs[:,:nTrain]
    fTest = fs[:,-nTest:]
    
    nxPts = xs.shape[0]
    nyPts = ys.shape[0]
    idx = np.arange(nxPts)
    idy = np.arange(nyPts)
    idxx, idyy = np.meshgrid(idx, idy)

    xs = xs[idyy.reshape(-1)]
    ys = ys[idxx.reshape(-1)]
    X = np.c_[xs, ys]
    Gref = LogCosKernel3D(xs, ys, n).reshape(nxPts, nyPts)

    h = 1 / fTrain.shape[0]
    uTrain = h * Gref @ fTrain 
    uTest = h * Gref @ fTest

    return fTrain, fTest, uTrain, uTest, X, Gref

def load_poisson2d_kernel_dataset_(
        data_root, nTrain, nTest, res=20, sigma='5e-01', sigma_test='2e-01'):
    
    mesh_path = os.path.join(data_root, 'mesh2D_disk_{:}.mat'.format(res))
    meshy_path = os.path.join(data_root, 'mesh2D_disk_{:}.mat'.format(res))
    fs_train_path = os.path.join(data_root, 'dat2D_disk_{:}_{:}.mat'.format(res, sigma))
    us_train_path = os.path.join(data_root, 'poisson2D_disk_{:}_{:}.mat'.format(res, sigma))
    fs_test_path = os.path.join(data_root, 'dat2D_disk_{:}_{:}.mat'.format(res, sigma_test))
    us_test_path = os.path.join(data_root, 'poisson2D_disk_{:}_{:}.mat'.format(res, sigma_test))

    print(f"load {mesh_path}")
    print(f"load {fs_train_path}")
    print(f"load {us_train_path}")
    print(f"load {fs_test_path}")
    print(f"load {us_test_path}")

    xs = scipy.io.loadmat(mesh_path)['X']
    ys = scipy.io.loadmat(meshy_path)['X']
    fs_train = scipy.io.loadmat(fs_train_path)['F']
    fs_test = scipy.io.loadmat(fs_test_path)['F']
    us_train = scipy.io.loadmat(us_train_path)['U']
    us_test = scipy.io.loadmat(us_test_path)['U']

    fTrain = fs_train[:,:nTrain]
    fTest = fs_test[:,-nTest:]
    uTrain = us_train[:,:nTrain]
    uTest = us_test[:,-nTest:]
    
    nxPts = xs.shape[0]
    nyPts = ys.shape[0]
    idx = np.arange(nxPts)
    idy = np.arange(nyPts)
    idxx, idyy = np.meshgrid(idx, idy)
    xxs = xs[idyy.reshape(-1)]
    yys = ys[idxx.reshape(-1)]
    X = np.c_[xxs, yys]
    Gref = None #PoissonKernelDisk(xxs, yys).reshape(nxPts, nyPts)

    return fTrain, fTest, uTrain, uTest, X, Gref

def load_helmholtz2d_kernel_dataset_(
        data_root, nTrain, nTest, res=20, sigma='5e-01', sigma_test='2e-01'):
    
    mesh_path = os.path.join(data_root, 'mesh2D_disk_{:}.mat'.format(res))
    meshy_path = os.path.join(data_root, 'mesh2D_disk_{:}.mat'.format(res))
    fs_train_path = os.path.join(data_root, 'dat2D_disk_{:}_{:}.mat'.format(res, sigma))
    us_train_path = os.path.join(data_root, 'helmholtz2D_disk_{:}_{:}.mat'.format(res, sigma))
    fs_test_path = os.path.join(data_root, 'dat2D_disk_{:}_{:}.mat'.format(res, sigma_test))
    us_test_path = os.path.join(data_root, 'helmholtz2D_disk_{:}_{:}.mat'.format(res, sigma_test))

    print(f"load {mesh_path}")
    print(f"load {fs_train_path}")
    print(f"load {us_train_path}")
    print(f"load {fs_test_path}")
    print(f"load {us_test_path}")

    xs = scipy.io.loadmat(mesh_path)['X']
    ys = scipy.io.loadmat(meshy_path)['X']
    fs_train = scipy.io.loadmat(fs_train_path)['F']
    fs_test = scipy.io.loadmat(fs_test_path)['F']
    us_train = scipy.io.loadmat(us_train_path)['U']
    us_test = scipy.io.loadmat(us_test_path)['U']

    fTrain = fs_train[:,:nTrain]
    fTest = fs_test[:,-nTest:]
    uTrain = us_train[:,:nTrain]
    uTest = us_test[:,-nTest:]
    
    nxPts = xs.shape[0]
    nyPts = ys.shape[0]
    idx = np.arange(nxPts)
    idy = np.arange(nyPts)
    idxx, idyy = np.meshgrid(idx, idy)
    xxs = xs[idyy.reshape(-1)]
    yys = ys[idxx.reshape(-1)]
    X = np.c_[xxs, yys]
    Gref = None

    return fTrain, fTest, uTrain, uTest, X, Gref


class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
