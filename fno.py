import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from neuralop.models import FNO
from utils import relative_err
from utils import init_records
from utils import UnitGaussianNormalizer

class MLP1d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP1d, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP2d, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class MLP3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP3d, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP1d(self.width, 1, self.width*2)  # output channel_dim is 1: u1(x)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic

        self.p = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP2d(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic

        self.p = nn.Linear(4, self.width)# input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)

        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q = MLP3d(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        return x


    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

class FNOFitter:
    def __init__(
            self,
            X, fs, us, fTest, uTest, bsz,
            device=None):
        
        '''
        X  : nPts x (nx + ny)
        fs : nPts x nSample 
        us : nPts x nSample 
        Gref : nPts x nPts
        '''

        # dataset 
        self.nyPts = fs.shape[0]
        self.nxPts = us.shape[0]
        self.inpDim = X.shape[1]
        self.X = X
        self.fs = fs
        self.us = us 
        self.fTest = fTest 
        self.uTest = uTest

        if self.inpDim == 2:
            self.fs = self.fs
            self.model = FNO1d(16, 64)
        elif self.inpDim == 4:
            self.model = FNO2d(12, 12, 32)
        elif self.inpDim == 6:
            self.model = FNO3d(8, 8, 8, 20)

        print()
        print("model structure")
        print(self.model)
        print()

        # log 
        self.Glog = []
        self.utest_log = []
        self.utrain_log = []
        
        # put to device 
        self.device = device 
        self.to_device()

        self.fNormalizer = UnitGaussianNormalizer(self.fs)
        self.fNormalizer.to(self.device)
        self.fs = self.fNormalizer.encode(self.fs)
        self.fTest = self.fNormalizer.encode(self.fTest)

        self.uNormalizer = UnitGaussianNormalizer(self.us)
        self.uNormalizer.to(self.device)
        self.us = self.uNormalizer.encode(self.us)
        

        # if self.inpDim == 6:
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.fs, self.us), batch_size=bsz, shuffle=True)


    def to_device(self):
        self.X = torch.from_numpy(self.X).float()
        self.fs = torch.from_numpy(self.fs).float()
        self.us = torch.from_numpy(self.us).float()
        self.fTest = torch.from_numpy(self.fTest).float()
        self.uTest = torch.from_numpy(self.uTest).float()

        self.X = self.X.to(self.device)
        self.fs = self.fs.to(self.device)
        self.us = self.us.to(self.device)
        self.fTest = self.fTest.to(self.device)
        self.uTest = self.uTest.to(self.device)
        self.model = self.model.to(self.device)
        
    # def optimize_adam(self, niter, lr=1e-3, step_size=100, gamma=0.5, dispstep=10):
    #     self.opt_adam = torch.optim.Adam(self.model.parameters(), lr)
    #     # self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_adam, T_max=niter)
    #     self.sch = torch.optim.lr_scheduler.StepLR(self.opt_adam, step_size=step_size, gamma=gamma)

    #     print()
    #     print("Adam optimization start")
    #     print()

    #     for k in range(niter):
    #         self.model.train()
    #         self.opt_adam.zero_grad()
    #         utrain_Pred = self.model(self.fs)
    #         utrain_Pred = self.uNormalizer.decode(utrain_Pred)
    #         utrain = self.uNormalizer.decode(utrain)
    #         loss = relative_err(utrain_Pred[...,0].reshape(-1,1).T, utrain[...,0].reshape(-1,1).T)
    #         loss.backward()
    #         self.opt_adam.step()
    #         self.sch.step()

    #         utrain_Pred = utrain_Pred[...,0].reshape(-1,1)
    #         us = us[...,0].reshape(-1,1)            
    #         utrain_rl2 = relative_err(utrain_Pred.detach().T.cpu().numpy(), us.T.cpu().numpy())

    #         self.model.eval()
    #         with torch.no_grad():
    #             utest_Pred = self.model(self.fTest)
    #             utest_Pred = self.uNormalizer.decode(utest_Pred)
    #             utest_Pred = utest_Pred[...,0].reshape(-1,1)
    #             uTest = self.uTest[...,0].reshape(-1,1)
    #             utest_rl2 = relative_err(utest_Pred.detach().T.cpu().numpy(), uTest.T.cpu().numpy())

    #         self.utest_log.append(utest_rl2)
    #         self.utrain_log.append(utrain_rl2)
            
    #         if k % dispstep == 0:
    #             print('{:}th train url2 : {:.4e} - test url2 : {:.4e}'.format(k, utrain_rl2, utest_rl2))

    #     # print('{:}th url2 : {:.4e} - Grl2 : {:.4e}'.format(k, utest_rl2))

    #     self.utest_Pred = utest_Pred.detach().cpu().numpy()
    #     self.log = {"utest_rl2" : np.array(self.utest_log), "utrain_rl2" : np.array(self.utrain_log)}
    
    def optimize_adam_batch(self, niter, lr=1e-3, step_size=100, gamma=0.5, dispstep=10):
        self.opt_adam = torch.optim.Adam(self.model.parameters(), lr)
        self.sch = torch.optim.lr_scheduler.StepLR(self.opt_adam, step_size=step_size, gamma=gamma)
        
        print()
        print("Adam optimization start")
        print()

        for k in range(niter):
            self.model.train()
            for ftrain, utrain in self.train_loader:
                ftrain, utrain = ftrain.to(self.device), utrain.to(self.device)
                self.opt_adam.zero_grad()
                utrain_Pred = self.model(ftrain)
                utrain_Pred = self.uNormalizer.decode(utrain_Pred)
                utrain = self.uNormalizer.decode(utrain)
                loss = relative_err(utrain_Pred[...,0].reshape(-1,1).T, utrain[...,0].reshape(-1,1).T)
                loss.backward()
                self.opt_adam.step()
            self.sch.step()

            self.model.eval()
            with torch.no_grad():
                utest_Pred = self.model(self.fTest)
                utest_Pred = self.uNormalizer.decode(utest_Pred)
                utest_Pred = utest_Pred[...,0].reshape(-1,1)
                uTest = self.uTest[...,0].reshape(-1,1)
                utest_rl2 = relative_err(utest_Pred.detach().T.cpu().numpy(), uTest.T.cpu().numpy())
                self.utest_log.append(utest_rl2)

            if k % dispstep == 0:
                print('{:}th train url2 : {:.4e} - {:}th test url2 : {:.4e} - learning rate : {:.4e}'.format(k, loss.item(), k, utest_rl2, self.sch.get_last_lr()[0]))

        # print('{:}th url2 : {:.4e} - Grl2 : {:.4e}'.format(k, utest_rl2))

        self.utest_Pred = utest_Pred.detach().cpu().numpy()
        self.log = {"utest_rl2" : np.array(self.utest_log), "utrain_rl2" : np.array(self.utrain_log)}
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGA for kernel estimation')
    parser.add_argument('--task', type=str, default='poisson1D',
                        help='dataset name. (poisson1D, helmholtz1D, airy1D, poisson2D)')
    parser.add_argument('--nIter', type=int, default=500, 
                        help='maximum number of neurons')
    parser.add_argument('--nTrain', type=int, default=200, 
                        help='number of training samples')
    parser.add_argument('--nTest', type=int, default=200, 
                        help='number of test samples')
    parser.add_argument('--res', type=int, default=20, 
                        help='mesh density')
    parser.add_argument('--sigma', type=str, default='2e-1', 
                        help='number of test samples')
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
    elif args.task == 'logcos3D':
        from utils import load_logcos3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_logcos3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, n=args.param, res=args.res, sigma=args.sigma)
    elif args.task == 'cos3D':
        from utils import load_cos3d_kernel_dataset
        fTrain, fTest, uTrain, uTest, X, Gref = load_cos3d_kernel_dataset(
            data_root='./data', nTrain=args.nTrain, nTest=args.nTest, n=args.param, res=args.res, sigma=args.sigma)
    
    fTrain = fTrain.T[...,None]
    uTrain = uTrain.T[...,None]
    fTest = fTest.T[...,None]
    uTest = uTest.T[...,None]

    if "3D" in args.task:
        fTrain = fTrain.reshape(-1,17,17,17,1)
        fTest = fTest.reshape(-1,17,17,17,1)
        uTrain = uTrain.reshape(-1,17,17,17,1)
        uTest = uTest.reshape(-1,17,17,17,1)

    model = FNOFitter(
        X = X,
        fs=fTrain, us=uTrain, 
        fTest=fTest, uTest=uTest, 
        bsz=20,
        device=device)

    # save outputs
    log_outpath, upred_outpath, model_outpath, Gpred_outpath = init_records('./results', args.task, 'fno-{:}-{:}-{:}-{:}'.format(args.nIter, args.nTrain, args.param, args.sigma))
    if os.path.exists(upred_outpath):
        print("********* ", args.task, "  ", upred_outpath, " file exists")
        exit()

    # model train
    if "3D" in args.task:
        model.optimize_adam_batch(args.nIter)
    # else:
    #     model.optimize_adam(args.nIter)

    np.save(log_outpath, model.log)
    np.save(upred_outpath, model.utest_Pred)