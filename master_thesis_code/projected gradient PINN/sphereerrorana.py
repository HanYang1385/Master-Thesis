# -*- coding: utf-8 -*-
import torch
from torch.nn import Linear
from torch.autograd import Variable
from torch import Tensor
import random
import math
import numpy as np
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# import os
# from google.colab import drive
# from google.colab import files
# import shutil

def set_seed_everywhere(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
set_seed_everywhere(666666)

###### sample by numpy, output tensor
def Sample_sphereunif(batch,Dim):
  ###### uniform sampling on the sphere
  mean = np.zeros(Dim)
  cov = np.eye(Dim)
  u0 = np.random.multivariate_normal(mean,cov,batch)
  u1 = u0 * u0
  u2 = np.sqrt(u1.sum(1))
  u3 = u2.reshape([batch,1])
  u4 = u3.repeat(Dim,1)
  u5 = u0 / u4
  tensor_u0 = torch.from_numpy(u5)
  tensor_u1 = tensor_u0.float()
  return tensor_u1

def Sample_batch(batch,Dim):
  ###### Dim = 3
  part1batch = int(batch)
  part2batch = int(0*batch)
  part1Xs = Sample_sphereunif(part1batch,Dim)
  part1Xr = Sample_sphereunif(part1batch,Dim)
  part2counterXs = Sample_sphereunif(part2batch,Dim)
  part2counterXr = -part2counterXs
  part1Xp = torch.cat((part1Xs,part1Xr),1)
  part2counterXp = torch.cat((part2counterXs,part2counterXr),1)
  Xp = torch.cat((part1Xp,part2counterXp),0)
  return Xp



###### Loss function from Eikonal equation
def EikonalLoss(Yobs, Xp, mu, W_s, Dim, batch, device):
    D_mu = torch.autograd.grad(outputs=mu, inputs=Xp, grad_outputs=torch.ones(mu.size()).to(device),
                               only_inputs=True, create_graph=True, retain_graph=True)[0]

    dmu = D_mu[:,Dim:2*Dim]
    batch_dmu = dmu.reshape(batch,Dim,1)

    ##### \boldsymbol{\vec{n}}(x) of \mathcal{S}
    Xr = Xp[:,Dim:2*Dim]
    Xr1 = Xr.reshape([batch,Dim])
    Xr2 = Xr1 * Xr1
    Xr2_norm = torch.sqrt(Xr2.sum(1))
    Xr3 = Xr2_norm.reshape([batch,1])
    Xr4 = Xr3.repeat([1,Dim])
    normal_x0 = Xr1 / Xr4
    normal_x = normal_x0.reshape(batch,Dim,1)

    ## 3rd-tensor
    Proj = torch.eye(Dim).repeat(batch,1,1) - torch.bmm(normal_x, torch.transpose(normal_x,1,2))
    batch_dmu_m = torch.bmm(Proj, batch_dmu)  
    dmu_m = batch_dmu_m.reshape(batch,Dim)

    Xrs = Xp[:,Dim:2*Dim] - Xp[:,0:Dim]
    Xrs2 = Xrs * Xrs
    Xrs2_norm = torch.sqrt(Xrs2.sum(1))
    Xrs3 = Xrs2_norm.reshape([batch,1])
    Xrs4 = Xrs3.repeat([1,Dim])
    normal_Xrs0 = Xrs / Xrs4
    normal_Xrs1 = normal_Xrs0.reshape(batch,1,Dim)

    # ###### Here can the eikonal equation can be reduced, so one can take the following two methods to calculate U1
    # ###### reduction $\big|\big| \nabla_\mathcal{S}^x||x-x_0|| \big|\big|^2 = \frac{1+x_0\cdot x}{2}$
    # Xs = Xp[:,0:Dim]
    # newtmp0 = Xs * Xr
    # newtmp1 = newtmp0.sum(1)
    # newU10 = (1 + newtmp1) / 2
    # mu2 = mu[:, 0] ** 2
    # U1 = newU10 * mu2

    ###### non-reduction $\big|\big| \nabla_\mathcal{S}^x||x-x_0|| \big|\big|^2 = \frac{1+x_0\cdot x}{2}$
    tmp1 = torch.bmm(normal_Xrs1,Proj)
    normal_Xrs2 = normal_Xrs0.reshape(batch,Dim,1)
    batch_U10 = torch.bmm(tmp1,normal_Xrs2)
    U10 = batch_U10.reshape(batch)
    mu2 = mu[:,0] ** 2
    U1 = U10 * mu2

    batch_dmu_t = dmu.reshape(batch,1,Dim)
    tmp2 = torch.bmm(Proj,batch_dmu)
    batch_U20 = torch.bmm(batch_dmu_t,tmp2)
    U20 = batch_U20.reshape(batch)
    U2 = Xrs2.sum(1) * U20

    batch_Xrs = Xrs.reshape(batch,1,Dim)
    batch_cross = torch.bmm(batch_Xrs,batch_dmu_m)
    cross = batch_cross.reshape(batch)
    U3 = 2 * mu[:,0] * cross
    
    Ypred0 = U1 + U2 + U3
    Ypred = Ypred0 * W_s

    numerator = (Ypred - Yobs) ** 2

    loss = torch.mean(numerator)

    return loss

###### weight initialization 
def init_weights(m):
    if type(m) == torch.nn.Linear:
        stdv = (1. / math.sqrt(m.weight.size(1))/1.)*2
        print('stdv is:\n',stdv)
        m.weight.data.uniform_(-stdv,stdv)
        m.bias.data.uniform_(-stdv,stdv)

###### architecture of neural network
class NN(torch.nn.Module):
    def __init__(self, nl=1, activation=torch.nn.ELU()):
        super(NN, self).__init__()
        self.act = activation

        # Input Structure
        self.fc0 = Linear(2*3, 64)

        # Resnet Block
        self.rn_fc1 = torch.nn.ModuleList([Linear(64,64) for i in range(nl)])
        self.rn_fc2 = torch.nn.ModuleList([Linear(64,64) for i in range(nl)])
        self.rn_fc3 = torch.nn.ModuleList([Linear(64,64) for i in range(nl)])

        # Output structure
        self.fc4 = Linear(64, 1)
        

    def forward(self, x):
        x = self.act(self.fc0(x))
        for ii in range(len(self.rn_fc1)):
            x0 = x
            x = self.act(self.rn_fc1[ii](x))
            x = self.act(self.rn_fc3[ii](x) + self.rn_fc2[ii](x0))

        mu = abs(self.fc4(x))
        return mu


###### initialize, save, load and train models
class Model():
    print('Enter Model!')
    def __init__(self, ModelPath, device='cpu'):
        self.Params = {}
        self.Params['ModelPath'] = ModelPath
        self.Params['Device'] = device
        self.Params['Pytorch Amp (bool)'] = False

        self.Params['Network'] = {}
        self.Params['Network']['Dimension'] = 3
        self.Params['Network']['Number of Residual Blocks'] = 1
        self.Params['Network']['Layer activation'] = torch.nn.ELU()

        self.Params['Training'] = {}
        self.Params['Training']['Batch Size'] = 128
        self.Params['Training']['Number of Iterations'] = 240000
        self.Params['Training']['Learning Rate'] = 5e-6
        self.Params['Training']['Use Scheduler (bool)'] = False

    def _init_network(self):
        self.network = NN(nl=self.Params['Network']['Number of Residual Blocks'],activation=self.Params['Network']['Layer activation'])
        self.network.apply(init_weights)
        self.network.float()
        self.network.to(torch.device(self.Params['Device']))

    def save(self,iteration='',distance_error=''):
      torch.save({'iteration':iteration,
                  'model_state_dict':self.network.state_dict(),
                  'optimizer_state_dict':self.optimizer.state_dict(),
                  'train_loss':self.list_train_loss,
                  'distance_error':self.list_distance_error,
                  'rel_distance_error':self.list_rel_distance_error,
                  'distance_error_counterpoint':self.list_distance_error_counterpoint,
                  'rel_distance_error_counterpoint':self.list_rel_distance_error_counterpoint},
                  '{}/Model_Iteration_{}_DistanceError_{}.pt'.format(self.Params['ModelPath'],str(iteration).zfill(5),distance_error))
    
    def load(self,load_filepath):
      self._init_network()
      checkpoint            = torch.load(load_filepath, map_location=torch.device(self.Params['Device']))
      self.network.load_state_dict(checkpoint['model_state_dict'])
      self.network.to(torch.device(self.Params['Device']))    
      
      self.list_train_loss = checkpoint['train_loss']
      self.list_distance_error         = checkpoint['distance_error']
      self.list_rel_distance_error       = checkpoint['rel_distance_error']
      self.list_distance_error_counterpoint         = checkpoint['distance_error_counterpoint']
      self.list_rel_distance_error_counterpoint       = checkpoint['rel_distance_error_counterpoint']

    def train(self):
        print('Enter training!')

        fix_myXs0 = Sample_sphereunif(1,self.Params['Network']['Dimension'])
        fix_myXs = fix_myXs0.reshape(self.Params['Network']['Dimension'])
        fix_myXr0 = Sample_sphereunif(1,self.Params['Network']['Dimension'])
        fix_myXr = fix_myXr0.reshape(self.Params['Network']['Dimension'])
        fix_myXp = torch.cat((fix_myXs,fix_myXr),0)

        ###### Initialising the network
        self._init_network()

        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=self.Params['Training']['Learning Rate'])
        if self.Params['Training']['Use Scheduler (bool)']:
          self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[50000,150000],gamma=0.1)

        self.list_train_loss = []
        self.list_distance_error = []
        self.list_rel_distance_error = []

        self.list_distance_error_counterpoint = []
        self.list_rel_distance_error_counterpoint = []

        for iteration in range(self.Params['Training']['Number of Iterations']):
            
            ####  sample a batch per 1000 steps
            if iteration % 1000 == 0:
              if (iteration>0) & (iteration%40000==0):
                self.Params['Training']['Batch Size'] = 2 * self.Params['Training']['Batch Size']
              Xp = Sample_batch(self.Params['Training']['Batch Size'],self.Params['Network']['Dimension'])


            Xp.requires_grad_()
            self.optimizer.zero_grad()

            output = self.network(Xp)
            W_r = torch.ones(self.Params['Training']['Batch Size'])
            W_s = torch.ones(self.Params['Training']['Batch Size'])
            loss = EikonalLoss(W_r, Xp, output, W_s, self.Params['Network']['Dimension'],
                               self.Params['Training']['Batch Size'], torch.device(self.Params['Device']))


            loss.backward()
            self.optimizer.step()

            train_loss = loss.item()
            self.list_train_loss.append(train_loss)

            if self.Params['Training']['Use Scheduler (bool)']:
              self.scheduler.step()
                       
            if iteration % 1000 == 0:

              ######### 指定起始点比较
              counterpoint_myXs0 = Tensor([[1,0,0]])
              counterpoint_norm_myXs0 = torch.sqrt(counterpoint_myXs0 @ counterpoint_myXs0.transpose(1,0))
              counterpoint_myXs1 = counterpoint_norm_myXs0.repeat([1,self.Params['Network']['Dimension']])
              counterpoint_myXs2 = counterpoint_myXs0 / counterpoint_myXs1
              counterpoint_myXs = counterpoint_myXs2.reshape([self.Params['Network']['Dimension']])
              # print('counterpoint_myXs is:',counterpoint_myXs)

              counterpoint_myXr0 = Tensor([[-1,0,0]])
              counterpoint_norm_myXr0 = torch.sqrt(counterpoint_myXr0 @ counterpoint_myXr0.transpose(1,0)) 
              counterpoint_myXr1 = counterpoint_norm_myXr0.repeat([1,self.Params['Network']['Dimension']])
              counterpoint_myXr2 = counterpoint_myXr0 / counterpoint_myXr1
              counterpoint_myXr = counterpoint_myXr2.reshape([self.Params['Network']['Dimension']])
              # print('counterpoint_myXr is:',counterpoint_myXr)
              counterpoint_myXp = torch.cat((counterpoint_myXs,counterpoint_myXr),0)
 
              ## Check!!! Counterpoint
              counterpoint_myXs2 = counterpoint_myXs * counterpoint_myXs
              counterpoint_norm_myXs = torch.sqrt(counterpoint_myXs2.sum(0))
              counterpoint_myXr2 = counterpoint_myXr * counterpoint_myXr
              counterpoint_norm_myXr = torch.sqrt(counterpoint_myXr2.sum(0))
              counterpoint_denominator = counterpoint_norm_myXs * counterpoint_norm_myXr
              counterpoint_numerator = counterpoint_myXs @ counterpoint_myXr
              counterpoint_tensor_real_arc = torch.arccos(counterpoint_numerator/counterpoint_denominator)
              counterpoint_real_arc = counterpoint_tensor_real_arc.item()
              print('counterpoint_real_arc is:',counterpoint_real_arc)

              counterpoint_myoutput = self.network(counterpoint_myXp)
              counterpoint_myXrs = counterpoint_myXr - counterpoint_myXs 
              counterpoint_myXrs2 = counterpoint_myXrs * counterpoint_myXrs
              counterpoint_norm_myXrs = torch.sqrt(counterpoint_myXrs2.sum(0))
              counterpoint_tensor_distance = counterpoint_myoutput * counterpoint_norm_myXrs ## 右端项非1时需要再乘以\sqrt(W(x_0))
              counterpoint_distance = counterpoint_tensor_distance.item()
              print('counterpoint_distance is:',counterpoint_distance)
              
              counterpoint_distance_error = counterpoint_real_arc - counterpoint_distance
              print('counterpoint_distance_error is:',counterpoint_distance_error)
              counterpoint_rel_distance_error = abs(counterpoint_distance_error) / counterpoint_real_arc
              print('counterpoint_rel_distance_error is:',counterpoint_rel_distance_error)
              self.list_distance_error_counterpoint.append(counterpoint_distance_error)
              self.list_rel_distance_error_counterpoint.append(counterpoint_rel_distance_error)


              ######### 随机取两点比较     
              myXs0 = Sample_sphereunif(1,self.Params['Network']['Dimension'])
              myXs = myXs0.reshape(self.Params['Network']['Dimension'])
              myXr0 = Sample_sphereunif(1,self.Params['Network']['Dimension'])
              myXr = myXr0.reshape(self.Params['Network']['Dimension'])
              myXp = torch.cat((myXs,myXr),0)
              # print('myXs is:',myXs)
              # print('myXr is:',myXr)

              ## Check!!! Fixed point
              fix_myXs2 = fix_myXs * fix_myXs
              fix_norm_myXs = torch.sqrt(fix_myXs2.sum(0))
              fix_myXr2 = fix_myXr * fix_myXr
              fix_norm_myXr = torch.sqrt(fix_myXr2.sum(0))
              fix_denominator = fix_norm_myXs * fix_norm_myXr
              fix_numerator = fix_myXs @ fix_myXr
              fix_tensor_real_arc = torch.arccos(fix_numerator/fix_denominator)
              fix_real_arc = fix_tensor_real_arc.item()
              print('fix_real_arc is:',fix_real_arc)

              fix_myoutput = self.network(fix_myXp)
              fix_myXrs = fix_myXr - fix_myXs 
              fix_myXrs2 = fix_myXrs * fix_myXrs
              fix_norm_myXrs = torch.sqrt(fix_myXrs2.sum(0))
              fix_tensor_distance = fix_myoutput * fix_norm_myXrs ## 右端项非1时需要再乘以\sqrt(W(x_0))
              fix_distance = fix_tensor_distance.item() * torch.sqrt(Tensor([0.5]))
              print('fix_distance is:',fix_distance)              

              ## Check!!! Random point
              myXs2 = myXs * myXs
              norm_myXs = torch.sqrt(myXs2.sum(0))
              myXr2 = myXr * myXr
              norm_myXr = torch.sqrt(myXr2.sum(0))
              denominator = norm_myXs * norm_myXr
              numerator = myXs @ myXr
              tensor_real_arc = torch.arccos(numerator/denominator)
              real_arc = tensor_real_arc.item()
              print('real_arc is:',real_arc)

              myoutput = self.network(myXp)
              myXrs = myXr - myXs 
              myXrs2 = myXrs * myXrs
              norm_myXrs = torch.sqrt(myXrs2.sum(0))
              tensor_distance = myoutput * norm_myXrs ## 右端项非1时需要再乘以\sqrt(W(x_0))
              distance = tensor_distance.item()
              print('distance is:',distance)              
              
              distance_error = real_arc - distance
              print('distance_error is:',distance_error)
              rel_distance_error = abs(distance_error) / real_arc
              print('rel_distance_error is:',rel_distance_error)
              self.list_distance_error.append(distance_error)
              self.list_rel_distance_error.append(rel_distance_error)

              print('learning rate is:',self.optimizer.state_dict()['param_groups'][0]['lr'])

              print('batch size is:',self.Params['Training']['Batch Size'])

              with torch.no_grad():
                print("iteration = {} -- Training loss = {:.4e} ".format(iteration+1, train_loss))
            
            if iteration == self.Params['Training']['Number of Iterations'] - 1:
              with torch.no_grad():
                self.save(iteration=iteration,distance_error=distance_error)
        
         
        return self.list_train_loss, self.list_distance_error, self.list_rel_distance_error, self.list_distance_error_counterpoint, self.list_rel_distance_error_counterpoint


###### get real solution of equation: $||\nabla_{\mathcal{S}_1}^x u(x;x_t)||=1$
def real_dis(xs,xr):
  xs2 = xs * xs
  norm_xs = torch.sqrt(xs2.sum(0))
  xr2 = xr * xr
  norm_xr = torch.sqrt(xr2.sum(0))
  denominator = norm_xs * norm_xr
  numerator = xs @ xr
  tensor_real_arc = torch.arccos(numerator/denominator)
  real_arc = tensor_real_arc.item()
  # print('real_arc is:',real_arc)
  return real_arc


###### get solution from neural network
def check(load_filepath,xs,xr,batch):
  ## xs, xr should be one dimension. xp = torch.cat((xs,xr),1) if they are two dimension. 
  checkpoint = torch.load(load_filepath)
  checkNN = NN(nl=1,activation=torch.nn.ELU())
  checkNN.load_state_dict(checkpoint['model_state_dict'])
  xp = torch.cat((xs,xr),1)
  check_output0 = checkNN(xp)
  # print('check_output0 is:\n',check_output0)
  check_output = check_output0.reshape(batch)
  # print('check_output.size() is:\n',check_output.size())
  xsr = xr - xs
  xsr2 = xsr * xsr
  # print('xsr2 is:\n',xsr2)
  norm_xsr = torch.sqrt(xsr2.sum(1))
  # print('norm_xsr is:\n',norm_xsr)
  tensor_check_distance = norm_xsr * check_output
  # check_distance = tensor_check_distance.item()
  return tensor_check_distance


###### sample on circle
def Samplecircle(batch,Dim,theta,R):
  fix = np.array([[0.0,0.0,1.0]])
  fix_batch = np.repeat(fix,batch,0)
  ###### rcalength = R * theta
  real_arc = R * theta
  ######
  r = R * np.sin(theta)
  z0 = R * np.cos(theta)
  z1 = z0.reshape([1,1])
  z_batch = np.repeat(z1,batch,0)
  uc0 = Sample_sphereunif(batch,Dim-1)
  uc1 = r * uc0
  s_batch = np.hstack((uc1,z_batch))
  # tensor_real_arc = torch.from_numpy(real_arc)
  # tensor_fix_batch = torch.from_numpy(fix_batch)
  # tensor_s_batch = torch.from_numpy(s_batch)
  # return real_arc, tensor_fix_batch.float(), tensor_s_batch.float()
  return real_arc, fix_batch, s_batch


###### sample moere points by orthogonal transformation on the sphere
def orthosample(fix_batch0,s_batch0,batch,Dim):
  fix_batch1 = fix_batch0.reshape(batch,1,Dim)
  s_batch1 = s_batch0.reshape(batch,1,Dim)
  ####### generate random orthogonal matrix
  orth0 = ortho_group.rvs(Dim)
  orth1 = orth0.reshape(1,Dim,Dim)
  orth_batch = orth1.repeat(batch,0)
  tmp_fix = np.einsum('ijk,ink->ijn',orth_batch,fix_batch1)
  tmp_s = np.einsum('ijk,ink->ijn',orth_batch,s_batch1)
  fix_batch = tmp_fix.reshape(batch,Dim)
  s_batch = tmp_s.reshape(batch,Dim)
  return s_batch, fix_batch
  

###### calculate error between neural network solution and real solution  
def error_distance(load_filepath,N,M,batch,Dim):
  list_rel_error = []
  theta = np.linspace(0,np.pi,N+1)
  for i in range(N+1):
    # print('theta[i] is:',theta[i])
    real_arc_i, fix_batch0_i, s_batch0_i = Samplecircle(batch,Dim,theta[i],1)
    ###### expectation on theta[i]
    list_rel_error_i = []
    for _ in range(M):
      spin_s_batch0, spin_fix_batch0 = orthosample(fix_batch0_i,s_batch0_i,batch,Dim)
      ###### convert array to tensor
      spin_fix_batch1 = torch.from_numpy(spin_fix_batch0)
      spin_fix_batch = spin_fix_batch1.float()
      spin_s_batch1 = torch.from_numpy(spin_s_batch0)
      spin_s_batch = spin_s_batch1.float()
      check_distance = check(load_filepath,spin_fix_batch,spin_s_batch,batch)
      abs_error_i = check_distance - real_arc_i
      rel_error_i = abs(abs_error_i) / real_arc_i
      mean_rel_error_i = torch.mean(rel_error_i)
      list_rel_error_i.append(mean_rel_error_i.item())
    # print('list_rel_error_i is:',list_rel_error_i)
    array_rel_error_i = np.array(list_rel_error_i)
    list_rel_error.append(array_rel_error_i.mean())
  return theta, list_rel_error



###### train model and save in filePath
filePath = '...\代码\投影梯度PINN'
model_sphereerrorana = Model(filePath)
list_train_loss_sphereerrorana, list_distance_error_sphereerrorana, list_rel_distance_error_sphereerrorana, list_distance_error_counterpoint_sphereerrorana, list_rel_distance_error_counterpoint_sphereerrorana = model_sphereerrorana.train()


###### calcaulate mean relate erroron sphere
###### reproduce results of paper by the well-trained model,
###### Model_Iteration_199999_DistanceError_0.07254266738891602.pt and Model_Iteration_239999_DistanceError_-0.003163456916809082.pt is the well-trained model
###### figure in appendix
# load_filepath = '...\代码\投影梯度PINN\Model_Iteration_199999_DistanceError_0.07254266738891602.pt'
##### figure in Penalty Weight PINN2
# load_filepath = '...\代码\投影梯度PINN\Model_Iteration_239999_DistanceError_-0.003163456916809082.pt'
# load_filepath = '...\代码\投影梯度PINN\Model_Iteration_239999_DistanceError_-0.0020682811737060547.pt'
# N = 150
# M = 60
# batch = 100
# Dim = 3
# theta, list_rel_error = error_distance(load_filepath,N,M,batch,Dim)
# print('list_rel_error is:\n',list_rel_error)
# # torch.save({'theta':theta,'list_rel_error':list_rel_error},'figure1.pt')
#
# ###### plot the figure of relate error on sphere
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=theta,y=list_rel_error,name="rel_Distance_Error"))
# fig.update_layout(width=700,height=480,template="plotly_white",margin=dict(l=5,r=5,t=5,b=5))
# fig.show()