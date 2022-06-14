import torch
from torch.nn import Linear
from torch.autograd import Variable
from torch import Tensor
import random
import math
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

###### 3 dimension discrete mesh
def mesh_on_simplex(M):
    lst = []
    ##### 分成几份就要写几个循环
    for i in range(M + 1):
        for j in range(M - i + 1):
            k = M - i - j
            if k >= 0:
                # print('(i,j,k) is:', i, j, k)
                lst.append([i, j, k])

    np_lst = np.array(lst) / M
    uni_np_lst = np.unique(np_lst, axis=0)
    return uni_np_lst

###### compute weight on discrete mesh
def mesh_compute_W(M,K):
    uni_np_lst = mesh_on_simplex(M)
    Num = uni_np_lst.shape[0]
    W = np.zeros([Num, Num])
    step_len = 1 / M * np.sqrt(2)
    for i in range(Num):
        for j in range(Num):
            criterion0 = uni_np_lst[i] - uni_np_lst[j]
            criterion1 = np.linalg.norm(criterion0)
            if criterion1 > 1.5 * step_len:
                W[i][j] = np.inf
            elif i == j:
                W[i][j] = 0
            else:
                W[i][j] = 1 * step_len
    return W


def startwith(start,wgraph):
    passed = [start]
    nopass = [x for x in range(len(wgraph)) if x != start]
    copy_wgraph = wgraph.copy()
    dis = copy_wgraph[start]

    while len(nopass):
        idx = nopass[0]
        ## first step from start point
        for i in nopass:
            if dis[i] < dis[idx]:
                idx = i
        nopass.remove(idx)
        passed.append(idx)
        for i in nopass:
            ###### 相当于记录从idx到终点的路径
            ###### 广度优先
            if dis[idx] + wgraph[idx][i] < dis[i]:
                dis[i] = dis[idx] + wgraph[idx][i]
    return dis


###### ||\nabla^x_\Omega u(x;x_0)|| = \sqrt{W(x)}
###### W(x) = 1+\frac{(x^\top\textbf{1}-1)^2}{\epsilon}

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed_everywhere(1234)

###### sample by numpy, output tensor
def Sample_on_simplex(batch,Dim):
    a = np.random.uniform(0,1,(batch,Dim-1))
    a_sorted = np.sort(a)
    extend_a = np.zeros((batch,Dim+1))
    extend_a[0:batch,1:Dim] = a_sorted
    extend_a[0:batch,-1] = 1
    minus_copy = np.zeros((batch,Dim+1))
    minus_copy[0:batch,2:Dim+1] = extend_a[0:batch,1:Dim]
    T0 = extend_a - minus_copy
    T = T0[0:batch,1:Dim+1]
    return T


def Sample_batch0(batch,Dim):
    hycub = np.random.uniform(0,1,(batch,Dim))
    return hycub


def Sample_batch(batch,Dim):
    numpy_Xs = Sample_batch0(batch,Dim)
    numpy_Xr = Sample_batch0(batch,Dim)
    numpy_Xp = np.hstack((numpy_Xs,numpy_Xr))
    ## np.random.shuffle(numpy_Xp)
    tensor_Xp0 = torch.from_numpy(numpy_Xp)
    tensor_Xp = tensor_Xp0.float()
    return tensor_Xp


def create_datasets(M, K):
    lst_on_simplex = mesh_on_simplex(M)
    len_lst = lst_on_simplex.shape[0]
    W = mesh_compute_W(M, K)

    fix_batch = 3 * (len_lst - 1)

    ###### start_point [0,0,1]
    start1 = 0
    fix_Xs1 = np.matlib.repmat(lst_on_simplex[0], len_lst - 1, 1)
    fix_Xr1 = lst_on_simplex[1:]
    dis1 = startwith(start1, W)
    ###### start_point [0,1,0]
    start2 = M
    fix_Xs2 = np.matlib.repmat(lst_on_simplex[M], len_lst - 1, 1)
    fix_Xr2 = np.vstack((lst_on_simplex[0:M], lst_on_simplex[M + 1:]))
    dis2 = startwith(start2, W)
    ###### start_point [1,0,0]
    start3 = len_lst - 1
    fix_Xs3 = np.matlib.repmat(lst_on_simplex[-1], len_lst - 1, 1)
    fix_Xr3 = lst_on_simplex[:-1]
    dis3 = startwith(start3, W)
    ###### Create Point Pairs
    np_fix_Xs = np.vstack((fix_Xs1, fix_Xs2, fix_Xs3))
    np_fix_Xr = np.vstack((fix_Xr1, fix_Xr2, fix_Xr3))
    tfix_Xs = torch.from_numpy(np_fix_Xs).float()
    tfix_Xr = torch.from_numpy(np_fix_Xr).float()
    np_fix_Xp = np.hstack((np_fix_Xs, np_fix_Xr))
    tfix_Xp = torch.from_numpy(np_fix_Xp).float()
    norm_fix_Xsr = torch.sqrt(((tfix_Xr - tfix_Xs) ** 2).sum(1)).reshape([fix_batch, 1])
    ###### Create real (numerical) value
    np_dis_totall = np.hstack((dis1[1:], dis2[0:M], dis2[M + 1:], dis3[:-1]))
    tdis_totall0 = torch.from_numpy(np_dis_totall).float()
    tdis_totall = tdis_totall0.reshape([fix_batch, 1])


    verify_batch = 3
    np_verify_Xs = np.array([np_fix_Xs[M - 1], np_fix_Xs[len_lst - 2], np_fix_Xs[2 * len_lst - 3]])
    tverify_Xs = torch.from_numpy(np_verify_Xs).float()
    np_verify_Xr = np.array([np_fix_Xr[M - 1], np_fix_Xr[len_lst - 2], np_fix_Xr[2 * len_lst - 3]])
    tverify_Xr = torch.from_numpy(np_verify_Xr).float()

    np_verify_Xp = np.array([np_fix_Xp[M - 1], np_fix_Xp[len_lst - 2], np_fix_Xp[2 * len_lst - 3]])
    tverify_Xp = torch.from_numpy(np_verify_Xp).float()
    norm_verify_Xsr = torch.sqrt(((tverify_Xr-tverify_Xs)**2).sum(1)).reshape([verify_batch,1])

    np_verify_dis = np.array([np_dis_totall[M - 1], np_dis_totall[len_lst - 2], np_dis_totall[2 * len_lst - 3]])
    tverify_dis = torch.from_numpy(np_verify_dis).float().reshape([3, 1])

    return tdis_totall, tfix_Xs, tfix_Xr, fix_batch, tfix_Xp, norm_fix_Xsr, tverify_Xs, tverify_Xr, verify_batch, tverify_Xp, norm_verify_Xsr, tverify_dis

###### Penalty can not be too large，i.e. \epsilon can not be too small. From the point of view of the loss function,
###### the orders of magnitude vary widely in different spatial locations, which means that training by SGD is difficult.
###### 右端项对非线性方程的影响是怎样的呢？


def compute_W_Penalty(K,Penalty_coef,X,batch,Dim):
    tensor_W = 1 + Penalty_coef * (X.sum(1)-1) ** 2
    return tensor_W

def compute_sqrtW_Penalty(K,Penalty_coef,X,batch,Dim):
    W_P = 1 + Penalty_coef * (X.sum(1)-1) ** 2
    tensor_sqrtW = torch.sqrt(W_P)
    return tensor_sqrtW

def FactoredEikonalLoss_expansionNoW(W_Pr, Xp, mu, W_Ps, Dim, batch, device):
    ###### detach() 不能乱用，用了相当于取消Xp.requires_grad_()；一般来说，神经网络的输出自动是requires_grad_()
    Xsr = Xp[:, Dim:2 * Dim] - Xp[:, 0:Dim]

    D_mu = torch.autograd.grad(outputs=mu, inputs=Xp, grad_outputs=torch.ones(mu.size()).to(device),
                               only_inputs=True, create_graph=True, retain_graph=True)[0]

    D_mu_r = D_mu[:, Dim:2 * Dim]
    item1 = (mu * mu).sum(1)
    Xsr2 = (Xsr * Xsr).sum(1)
    D_mu_r2 = (D_mu_r * D_mu_r).sum(1)
    item2 = Xsr2 * D_mu_r2
    item30 = torch.einsum("ij,ij->i", Xsr, D_mu_r)
    item3 = 2 * mu.reshape(batch) * item30

    lhs2 = item1 + item2 + item3
    rhs2 = W_Pr
    SE = (lhs2 - rhs2) ** 2
    MSE = torch.mean(SE)
    return MSE


def init_weights(m):
    if type(m) == torch.nn.Linear:
        stdv = (1. / math.sqrt(m.weight.size(1)) / 1.) * 2
        print('stdv is:\n', stdv)
        m.weight.data.uniform_(-stdv, stdv)
        m.bias.data.uniform_(-stdv, stdv)


class NN(torch.nn.Module):
    def __init__(self, nl=1, activation=torch.nn.ELU()):
        super(NN, self).__init__()
        self.act = activation

        # Input Structure
        self.fc0 = Linear(2 * 3, 64)

        # Resnet Block
        self.rn_fc1 = torch.nn.ModuleList([Linear(64, 64) for i in range(nl)])
        self.rn_fc2 = torch.nn.ModuleList([Linear(64, 64) for i in range(nl)])
        self.rn_fc3 = torch.nn.ModuleList([Linear(64, 64) for i in range(nl)])

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
        self.Params['Training']['mesh'] = 40
        self.Params['Training']['Batch Size'] = 128
        self.Params['Training']['Number of Iterations'] = 400000  # 240000
        self.Params['Training']['Learning Rate'] = 0.0005
        self.Params['Training']['Use Scheduler (bool)'] = False
        self.Params['Training']['Penalty'] = 20
        self.Params['Training']['K'] = 1

    def _init_network(self):
        self.network = NN(nl=self.Params['Network']['Number of Residual Blocks'],
                          activation=self.Params['Network']['Layer activation'])
        self.network.apply(init_weights)
        self.network.float()
        self.network.to(torch.device(self.Params['Device']))

    def save(self, iteration=''):
        torch.save({'iteration': iteration,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': self.list_train_loss,
                    'relate_mean_error':self.list_fix_mean_rel_error},
                    '{}/Model_Iteration_{}_loss_{}.pt'.format(self.Params['ModelPath'], str(iteration).zfill(5),
                                                             self.list_train_loss[-1]))

    def train(self):
        print('Enter training!')
        print('Penalty_Coef is:',self.Params['Training']['Penalty'])
        ###### Create Verify Dataset
        tdis_totall, tfix_Xs, tfix_Xr, fix_batch, tfix_Xp, norm_fix_Xsr, tverify_Xs, tverify_Xr, verify_batch, tverify_Xp, norm_verify_Xsr, tverify_dis = create_datasets(self.Params['Training']['mesh'],self.Params['Training']['K'])

        fix_sqrtW_s = compute_sqrtW_Penalty(self.Params['Training']['K'],self.Params['Training']['Penalty'],tfix_Xs,fix_batch,self.Params['Network']['Dimension'])
        verify_sqrtW_s = compute_sqrtW_Penalty(self.Params['Training']['K'],self.Params['Training']['Penalty'],tverify_Xs,verify_batch,self.Params['Network']['Dimension'])

        ###### Initialising the network
        self._init_network()

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.Params['Training']['Learning Rate'])
        if self.Params['Training']['Use Scheduler (bool)']:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150000, 300000], gamma=0.1)

        self.list_train_loss = []
        self.list_fix_mean_rel_error = [] ###### the total mesh points
        self.list_verify_mean_rel_error = []

        for iteration in range(self.Params['Training']['Number of Iterations']):
            ####  sample a batch per 1000 steps
            if iteration % 1000 == 0:
                if (iteration > 0) & (iteration % 80000 == 0):
                    self.Params['Training']['Batch Size'] = 2 * self.Params['Training']['Batch Size']
                Xp = Sample_batch(self.Params['Training']['Batch Size'], self.Params['Network']['Dimension'])

            Xp.requires_grad_()
            [Xs, Xr] = torch.split(Xp, self.Params['Network']['Dimension'], 1)
            self.optimizer.zero_grad()

            output = self.network(Xp)
            W_Ps = compute_W_Penalty(self.Params['Training']['K'],self.Params['Training']['Penalty'],Xs,self.Params['Training']['Batch Size'],self.Params['Network']['Dimension'])
            W_Pr = compute_W_Penalty(self.Params['Training']['K'],self.Params['Training']['Penalty'],Xr,self.Params['Training']['Batch Size'],self.Params['Network']['Dimension'])
            loss_expansion = FactoredEikonalLoss_expansionNoW(W_Pr,Xp,output,W_Ps,self.Params['Network']['Dimension'],self.Params['Training']['Batch Size'],torch.device(self.Params['Device']))

            loss_expansion.backward()
            self.optimizer.step()

            train_loss = loss_expansion.item()
            self.list_train_loss.append(train_loss)

            if self.Params['Training']['Use Scheduler (bool)']:
                self.scheduler.step()

            if iteration % 1000 == 0:
                fix_output = self.network(tfix_Xp)
                verify_output = self.network(tverify_Xp)

                tfix_u = norm_fix_Xsr * fix_output
                print('tfix_u is:\n', tfix_u)

                tverify_u = norm_verify_Xsr * verify_output
                print('tverify_u is:\n', tverify_u)

                fix_real_sol = norm_fix_Xsr
                fix_my_sol = tfix_u
                fix_abs_error = abs(fix_real_sol-fix_my_sol)
                fix_relate_error = fix_abs_error / fix_real_sol
                print('fix_relate_error is:\n',fix_relate_error)
                self.list_fix_mean_rel_error.append(torch.mean(fix_relate_error).item())
                print('torch.mean(fix_relate_error).item() is:',torch.mean(fix_relate_error).item())

                verify_abs_error = abs(norm_verify_Xsr-tverify_u)
                verify_relate_error = verify_abs_error / norm_verify_Xsr
                print('verify_relate_error is:\n',verify_relate_error)
                self.list_verify_mean_rel_error.append(torch.mean(verify_relate_error).item())
                print('torch.mean(verify_relate_error).item() is:',torch.mean(verify_relate_error).item())


                print('learning rate is:', self.optimizer.state_dict()['param_groups'][0]['lr'])
                print('batch size is:', self.Params['Training']['Batch Size'])

                with torch.no_grad():
                    print("iteration = {} -- Training loss = {:.4e} ".format(iteration + 1, train_loss))

            if iteration == self.Params['Training']['Number of Iterations'] - 1:
                with torch.no_grad():
                    self.save(iteration=iteration)

        return self.list_train_loss, self.list_fix_mean_rel_error, self.list_verify_mean_rel_error

def load(load_filepath,learning_rate):
    loadpoint = torch.load(load_filepath)
    loadNN = NN(nl=1,activation=torch.nn.ELU())
    loadNN.load_state_dict(loadpoint['model_state_dict'])
    loadNN.to(torch.device('cpu'))
    # ## keep parameters for optimizer
    # loadOpt = torch.optim.Adam(loadNN.parameters(),lr=learning_rate)
    # loadOpt.load_state_dict(loadpoint['optimizer_state_dict'])
    # ## using new learning rate for optimizer
    loadOpt = torch.optim.Adam(loadNN.parameters(),lr=learning_rate)
    train_loss = loadpoint['train_loss']


def check(load_filepath,Xs,Xr,batch,K,Penalty_coef,Dim):
    Xp = torch.cat((Xs,Xr),1)
    norm_Xsr = torch.sqrt(((Xr-Xs)**2).sum(1))

    checkpoint = torch.load(load_filepath)
    checkNN = NN(nl=1,activation=torch.nn.ELU())
    checkNN.load_state_dict(checkpoint['model_state_dict'])
    check_output0 = checkNN(Xp)
    check_output = check_output0.reshape(batch)
    tensor_check = norm_Xsr * check_output
    return tensor_check

def save(network,optimizer,list_train_loss,list_mean_rel_error,ModelPath,iteration=''):
    torch.save({'iteration': iteration,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': list_train_loss,
                'relate_mean_error': list_mean_rel_error},
                '{}/Model_Iteration_{}_loss_{}.pt'.format(ModelPath, str(iteration).zfill(5), list_train_loss[-1]))

def continue_train(load_filepath,continu_trainpath,learning_rate,BSize,Dim,Iters,K,Penalty_coef,M):
    print('Enter continue training!')
    ###### Create Verify Dataset

    tdis_totall, tfix_Xs, tfix_Xr, fix_batch, tfix_Xp, norm_fix_Xsr = create_datasets(M,K)
    fix_sqrt_Ws = compute_sqrtW_Penalty(K,Penalty_coef,tfix_Xs,fix_batch,Dim)
    continupoint = torch.load(load_filepath)
    continuNN = NN(nl=1, activation=torch.nn.ELU())
    continuNN.load_state_dict(continupoint['model_state_dict'])
    # ## keep optimizer's parameters
    # continuOpt = torch.optim.Adam(continuNN.parameters(), lr=learning_rate)
    # continuOpt.load_state_dict(continupoint['optimizer_state_dict'])
    # ## using new learning rate for optimizer
    continuOpt = torch.optim.Adam(continuNN.parameters(), lr=learning_rate)
    list_continue_train_loss = []
    list_continue_mean_rel_error = []
    for iteration in range(Iters):
        if iteration % 1000 == 0:
            if (iteration > 0) & (iteration % 40000 == 0):
                BSize = 2 * BSize
            Xp = Sample_batch(BSize,Dim)

        Xp.requires_grad_()
        [Xs, Xr] = torch.split(Xp,Dim, 1)
        continuOpt.zero_grad()
        output = continuNN(Xp)
        W_Ps = compute_W_Penalty(K,Penalty_coef,Xs,BSize,Dim)
        W_Pr = compute_W_Penalty(K,Penalty_coef,Xr,BSize,Dim)
        loss_expansion = FactoredEikonalLoss_expansionNoW(W_Pr,Xp,output,W_Ps,Dim,BSize,torch.device('cpu'))
        loss_expansion.backward()
        continuOpt.step()

        contin_train_loss = loss_expansion.item()
        list_continue_train_loss.append(contin_train_loss)

        if iteration % 1000 == 0:
            fix_output = continuNN(tfix_Xp)
            tensor_u = norm_fix_Xsr * fix_output
            real_sol = tdis_totall
            my_sol = tensor_u
            relate_error = abs(real_sol-my_sol) / real_sol
            list_continue_mean_rel_error.append(torch.mean(relate_error).item())
            print('verify is:', torch.mean(relate_error).item())
            print('learning rate is:', continuOpt.state_dict()['param_groups'][0]['lr'])
            print('batch size is:', BSize)

            with torch.no_grad():
                print("iteration = {} -- Continue Training loss = {:.4e} ".format(iteration + 1, contin_train_loss))

        if iteration == Iters - 1:
            with torch.no_grad():
                save(continuNN,continuOpt,list_continue_train_loss,list_continue_mean_rel_error,continu_trainpath,iteration=iteration)

    return list_continue_train_loss, list_continue_mean_rel_error, tdis_totall, tfix_Xp, norm_fix_Xsr




###### Training Model
model_filePath = '...\代码\权重惩罚PINN1'
model_FacEikNet_Penalty_expansion = Model(model_filePath)
list_train_loss_EikNet_Penalty_expansion, list_fix_mean_rel_error, list_verify_mean_rel_error = model_FacEikNet_Penalty_expansion.train()
plt.plot(list_fix_mean_rel_error)
plt.show()

# ###### Continue Training Model
# continu_trainpath = '...'
# learning_rate = 1e-7
# BSize = 2048
# Dim = 3
# Iters = 80000
# Penalty_coef = 10
# M = 100
# list_continue_train_loss, list_continue_mean_rel_error, tdis_totall, tfix_Xp, norm_fix_Xsr = continue_train(load_filepath,continu_trainpath,learning_rate,BSize,Dim,Iters,K,Penalty_coef,M)


# ###### Check
# load_filepath = '...\代码\权重惩罚PINN1\文章中实验一训练好的神经网络\Model_Iteration_..._loss_....pt'
# Dim = 3
# M = 80
# print('M is:',M)
# K = 1
# Penalty_coef = 25
# print('Penalty_coef is:',Penalty_coef)
# Dim = 3
#
# tdis_totall, tfix_Xs, tfix_Xr, fix_batch, tfix_Xp, norm_fix_Xsr, tverify_Xs, tverify_Xr, verify_batch, tverify_Xp, norm_verify_Xsr, tverify_dis = create_datasets(M,K)
#
# ## verify1：average of all points
# fix_batch = tfix_Xp.shape[0]
# tensor_fixcheck = check(load_filepath,tfix_Xs,tfix_Xr,fix_batch,K,Penalty_coef,Dim)
#
# ###### transform Tensor to numpy in order to analysis
# np_dis_totall = tdis_totall.numpy().reshape([fix_batch,1])
# np_fixcheck = tensor_fixcheck.detach().numpy().reshape([fix_batch,1])
# np_norm_fix_Xsr = norm_fix_Xsr.numpy()
# fix_rel_error = abs(np_norm_fix_Xsr-np_fixcheck) / np_norm_fix_Xsr
#
#
# ## verify2：average between vertexs
# verify_batch = tverify_Xp.shape[0]
# tensor_verifycheck = check(load_filepath,tverify_Xs,tverify_Xr,verify_batch,K,Penalty_coef,Dim)
#
# ###### transform Tensor to numpy in order to analysis
# np_verify_dis = tverify_dis.numpy().reshape([verify_batch,1])
# np_verifycheck = tensor_verifycheck.detach().numpy().reshape([verify_batch,1])
# np_norm_verify_Xsr = norm_verify_Xsr.numpy()
# verify_rel_error = abs(np_norm_verify_Xsr - np_verifycheck) / np_norm_verify_Xsr
# print('verify_rel_error.mean() is:', verify_rel_error.mean())
# print('fix_rel_error.mean() is:', fix_rel_error.mean())


