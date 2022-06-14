import torch
from torch.nn import Linear
from torch.autograd import Variable
from torch import Tensor
import random
import math
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


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
                W[i][j] = step_len * np.sqrt(uni_np_lst[j] @ K @ uni_np_lst[j])
                # W[i][j] = step_len * np.sqrt( uni_np_lst[i] @ K @ uni_np_lst[i] * (1 + 10*(uni_np_lst[i].sum() - 1)**2) )
    return W


def startwith(start,wgraph):
    passed = [start]
    nopass = [x for x in range(len(wgraph)) if x != start]
    copy_wgraph = wgraph.copy()
    dis = copy_wgraph[start]
    # print('passed is:',passed)
    # print('nopass is:',nopass)
    # print('dis is:',dis)

    while len(nopass):
        idx = nopass[0]
        # print('1 idx is:',idx)
        # print('passed is:',passed)
        # print('nopass is:',nopass)
        ## first step from start point
        for i in nopass:
            if dis[i] < dis[idx]:
                idx = i
        nopass.remove(idx)
        passed.append(idx)
        # print('dis is:',dis)
        for i in nopass:
            ###### 相当于记录从idx到终点的路径
            ###### 广度优先
            if dis[idx] + wgraph[idx][i] < dis[i]:
                dis[i] = dis[idx] + wgraph[idx][i]
    return dis

M = 20
## M = 20, 对应3个顶点的编号为：0,20,230
## M = 40, 对应的3个顶点的编号为：0,40,860
# Flory Huggins parameters come from Affinity matrix
# K = np.array([[1.1933,1.2691,1.3207],[1.2691,1.3184,1.3974],[1.3207,1.3974,1.5172]])
# K = np.array([[11.9331,12.6915,13.2069],[12.6915,13.1843,13.9742],[13.2069,13.9742,15.1716]])
K = np.array([[1193.31,1269.15,1320.69],[1269.15,1318.43,1397.42],[1320.69,1397.42,1517.16]])
lst_on_simplex = mesh_on_simplex(M)
W = mesh_compute_W(M,K)

start1 = 0
dis1 = startwith(start1,W)

start2 = 20
dis2 = startwith(start2,W)

start3 = 230
dis3 = startwith(start3,W)

###### matrix sig is "surface tension"
sig = np.array([[dis1[0],dis1[20],dis1[-1]],[dis2[0],dis2[20],dis2[-1]],[dis3[0],dis3[20],dis3[-1]]])



