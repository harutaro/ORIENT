# cudaプログラム(C++) との結合方法： python -m setup install
#
# 2025/03/23 sow_cpp.forward_V で Rank Control を導入
# 2025/03/23 sow_cpp.backard_V で Rank Control を導入
# 2025/04/18 sow_cpp.forward_Uh で Rank Control を導入
# 2025/04/18 sow_cpp.backard_Uh で Rank Control を導入
# 2025/05/02 sow_cpp.forward_S で Rank Control を導入中
# 2025/05/02 sow_cpp.backard_S で Rank Control を導入中

import os
import sys
import time
import numpy as np
import random
import math
#import pynvml

import torch
import torch.nn as nn
from torch.autograd import Function
import sow_cpp

def print_ptbl(tbl):
    N = int((1+math.sqrt(1+4*tbl.shape[0]))/2)
    print(f'N={N}')
    cnt = 0
    for i in range(N-1):
        if N > 16 :
            if i < 2 or i > N-3:
                print(f'(%4d) ptbl[%7d:%7d]: (%4d, %4d), (%4d, %4d), ..., (%4d, %4d)'
                      % (i, cnt, cnt+N, tbl[cnt], tbl[cnt+1],
                         tbl[cnt+2], tbl[cnt+3],
                         tbl[cnt+N-2], tbl[cnt+N-1]))
        else :
            print(f'(%2d) ptbl[%2d:%2d]:' % (i, cnt, cnt+N), end='')
            for j in range(int(N/2)):
                print(f' (%2d,%2d)' % (tbl[cnt+2*j], tbl[cnt+2*j+1]), end='')
                if j < int(N/2)-1 : print(',', end='')
            print(']')
        cnt += N
    return

def print_pidx(tbl):
    N = int((1+math.sqrt(1+4*tbl.shape[0]))/2)
    print(f'N={N}')
    cnt = 0
    for i in range(N-1):
        if N > 16 :
            if i < 2 or i > N-3:
                i1 = tbl[cnt]
                j1 = tbl[cnt+1]
                i2 = tbl[cnt+2]
                j2 = tbl[cnt+3]
                i3 = tbl[cnt+N-2]
                j3 = tbl[cnt+N-1]
                idx1 = int(i1 * (2 * N - 3 - i1) / 2) + j1 - 1
                idx2 = int(i2 * (2 * N - 3 - i2) / 2) + j2 - 1
                idx3 = int(i3 * (2 * N - 3 - i3) / 2) + j3 - 1
                print(f'(%4d) pidx[%7d:%7d]:  %4d, %4d, ..., %4d'
                      % (i, cnt, cnt+N, idx1, idx2, idx3))
        else :
            print(f'(%2d) pidx[%2d:%2d]:' % (i, cnt, cnt+N), end='')
            for j in range(int(N/2)):
                ii = tbl[cnt+2*j]
                jj = tbl[cnt+2*j+1]
                idx = int(ii * (2 * N - 3 - ii) / 2) + jj - 1
                print(f' %2d' % (idx), end='')
                if j < int(N/2)-1 : print(',', end='')
            print('')
        cnt += N
    return

############################################################
''' https://pytorch.org/docs/stable/notes/extending.html '''

class V_Function(Function):
    """Output & Gradient Calcuation for Z1=X*V """
    @staticmethod
    def forward(ctx, Z0, tan_2theta, ptbl, rank, freezed_rank):
        #print('[V_Function forward]')
        BS = int(Z0.shape[0])
        N = int(Z0.shape[1])
        rank_T = torch.tensor(rank, dtype=torch.int, device=Z0.device, requires_grad=False)
        freezed_rank_T = torch.tensor(freezed_rank, dtype=torch.int, device=Z0.device, requires_grad=False)
        rid_T = torch.tensor(19, dtype=torch.int, device=Z0.device, requires_grad=False)
        cos_theta = torch.empty((1, int(N*(N-1)/2)), dtype=Z0.dtype, device=Z0.device, requires_grad=False)
        sin_theta = torch.empty((1, int(N*(N-1)/2)), dtype=Z0.dtype, device=Z0.device, requires_grad=False)
        ZZ = torch.empty((1, N*N*BS), dtype=Z0.dtype, device=Z0.device, requires_grad=False)
        Z1 = torch.empty((BS, N), dtype=Z0.dtype, device=Z0.device, requires_grad=True)
        sow_cpp.forward_V(Z0, ptbl, rank_T, tan_2theta, cos_theta, sin_theta, ZZ, Z1)

        '''
        len_in = float(torch.sum(torch.diag(Z0.mm(Z0.t()))))
        len_out = float(torch.sum(torch.diag(Z1.mm(Z1.t()))))
        if abs((len_in - len_out) / len_in) > - 1.0e-15 :
            print(f'BS={BS}, N={N}, R={r}')
            print('ptbl:', ptbl.shape, ptbl.dtype, ptbl.device, ptbl.requires_grad)
            print_ptbl(ptbl)
            print('tan_2theta:', tan_2theta.shape, tan_2theta.device, tan_2theta.dtype, tan_2theta.requires_grad)
            print(tan_2theta)
            print('Z0:', Z0.shape, Z0.device, Z0.dtype, Z0.requires_grad)
            print(Z0)
            print('Z1:', Z1.shape, Z1.device, Z1.dtype, Z1.requires_grad)
            print(Z1)
            #print(torch.diag(Z0.mm(Z0.t())))
            #print(torch.diag(Z1.mm(Z1.t())))
            print(f'abs(|Z0|-|Z1|) / |Z0| = ||%.3e| - |%.3e|| / |%.3e| = %.3e' %
                  (len_in, len_out, len_in, abs((len_in - len_out) / len_in)))
            tmpV = torch.empty((N, N), dtype=Z0.dtype, device=Z0.device, requires_grad=False)
            sow_cpp.setup_V(ptbl, cos_theta, sin_theta, tmpV);
            Y2 = Z0.mm(tmpV)
            #print('tmpV:', tmpV.shape, tmpV.device, tmpV.dtype, tmpV.requires_grad)
            #print(tmpV)
            #print('Y2:', Y2.shape, Y2.device, Y2.requires_grad)
            #print(Y2)
            print(f'|Z1-Y2| / N = %.3e' % (float(torch.linalg.norm(Z1 - Y2))/N))
            #exit()
        '''

        ctx.save_for_backward(tan_2theta, cos_theta, sin_theta, ZZ, ptbl, rank_T, freezed_rank_T)
        return Z1

    @staticmethod
    def backward(ctx, grad_Z1):
        #print('[V_Function backward]')
        tan_2theta, cos_theta, sin_theta, ZZ, ptbl, rank_T, freezed_rank_T = ctx.saved_tensors
        BS = int(grad_Z1.shape[0])
        N = int(grad_Z1.shape[1])
        grad_Z0 = torch.empty((BS, N), dtype=grad_Z1.dtype, device=grad_Z1.device, requires_grad=False)
        grad_tan_2theta = torch.empty((BS, int(N*(N-1)/2)),
                                      dtype=grad_Z1.dtype, device=grad_Z1.device, requires_grad=False)
        sow_cpp.backward_V(grad_Z1, ptbl, rank_T, freezed_rank_T, tan_2theta,
                           cos_theta, sin_theta, ZZ, grad_Z0, grad_tan_2theta)
        # 全サンプル-mean方式（→性能劣化を招くので全サンプル使用の方がよさそう → 次行をコメントアウト）
        #grad_tan_2theta = grad_tan_2theta.mean(axis=0).view(1,-1)

        '''
        len_in = float(torch.sum(torch.diag(grad_Z1.mm(grad_Z1.t()))))
        len_out = float(torch.sum(torch.diag(grad_Z0.mm(grad_Z0.t()))))
        if abs((len_in - len_out) / len_in) > -1.0e-15 :
            print(f'BS={BS}, N={N}')
            print('ptbl:', ptbl.shape, ptbl.dtype, ptbl.device, ptbl.requires_grad)
            print_ptbl(ptbl)
            print('tan_2theta:', tan_2theta.shape, tan_2theta.dtype, tan_2theta.device, tan_2theta.requires_grad)
            print(tan_2theta)
            print('grad_Z1:', grad_Z1.shape, grad_Z1.dtype, grad_Z1.device, grad_Z1.requires_grad)
            print(grad_Z1)
            print('grad_Z0:', grad_Z0.shape, grad_Z0.dtype, grad_Z0.device, grad_Z0.requires_grad)
            print(grad_Z0)
            #print(torch.diag(grad_Z1.mm(grad_Z1.t())))
            #print(torch.diag(grad_Z0.mm(grad_Z0.t())))
            print(f'abs((|grad_Z1|-|grad_Z0|)/|Z0|) = (|%.3e| - |%.3e|) / |%.3e| = %.3e' %
                  (len_in, len_out, len_in, abs((len_in - len_out) / len_in)))
            print('grad_tan_2theta:', grad_tan_2theta.shape, grad_tan_2theta.dtype, grad_tan_2theta.device, grad_tan_2theta.requires_grad)
            print(grad_tan_2theta)
            #exit()
        '''

        #print(' ctx.needs_input_grad:',  ctx.needs_input_grad)
        if ctx.needs_input_grad[0] == False : grad_Z0 = None
        if ctx.needs_input_grad[1] == False : grad_tan_2theta = None
        # ptbl, rid, ttbl has no grad.
        return grad_Z0, grad_tan_2theta, None, None, None, None, None

class U_Function(Function):
    """Output & Gradient Calcuation for Y=Z2*(U^T) """
    @staticmethod
    def forward(ctx, Z2, tan_2phi, ptbl, rank, freezed_rank):
        #print('[U_Function forward]')
        BS = int(Z2.shape[0])
        N = int(Z2.shape[1])
        rank_T = torch.tensor(rank, dtype=torch.int, device=Z2.device, requires_grad=False)
        freezed_rank_T = torch.tensor(freezed_rank, dtype=torch.int, device=Z2.device, requires_grad=False)
        cos_phi = torch.empty((1, int(N*(N-1)/2)), dtype=Z2.dtype, device=Z2.device, requires_grad=False)
        sin_phi = torch.empty((1, int(N*(N-1)/2)), dtype=Z2.dtype, device=Z2.device, requires_grad=False)
        ZZ = torch.empty((1, N*N*BS), dtype=Z2.dtype, device=Z2.device, requires_grad=False)
        Y = torch.empty((BS, N), dtype=Z2.dtype, device=Z2.device, requires_grad=True)
        sow_cpp.forward_Uh(Z2, ptbl, rank_T, tan_2phi, cos_phi, sin_phi, ZZ, Y)

        '''
        len_in = float(torch.sum(torch.diag(Z2.mm(Z2.t()))))
        len_out = float(torch.sum(torch.diag(Y.mm(Y.t()))))
        if abs((len_in - len_out) / len_in) > -1.0e-15 :
            print(f'BS={BS}, N={N}, R={r}')
            print('ptbl:', ptbl.shape, ptbl.dtype, ptbl.device, ptbl.requires_grad)
            print_ptbl(ptbl)
            print('tan_2phi:', tan_2phi.shape, tan_2phi.device, tan_2phi.dtype, tan_2phi.requires_grad)
            print(tan_2phi)
            print('Z2:', Z2.shape, Z2.device, Z2.dtype, Z2.requires_grad)
            print(Z2)
            print('Y:', Y.shape, Y.device, Y.dtype, Y.requires_grad)
            print(Y)
            #print(torch.diag(Z2.mm(Z2.t())))
            #print(torch.diag(Y.mm(Y.t())))
            print(f'abs(|Z2|-|Y|) / |Z0| = ||%.3e| - |%.3e|| / |%.3e| = %.3e' %
                  (len_in, len_out, len_in, abs((len_in - len_out) / len_in)))
            tmpUh = torch.empty((N, N), dtype=Z2.dtype, device=Z2.device, requires_grad=False)
            sow_cpp.setup_Uh(ptbl, cos_phi, sin_phi, tmpUh);
            Y2 = Z2.mm(tmpUh)
            #print('tmpUh:', tmpUh.shape, tmpUh.device, tmpUh.dtype, tmpUh.requires_grad)
            #print(tmpUh)
            #print('Y2:', Y2.shape, Y2.device, Y2.requires_grad)
            #print(Y2)
            print(f'|Y-Y2| / N = %.3e' % (float(torch.linalg.norm(Y - Y2))/N))
            #exit()
        '''

        ctx.save_for_backward(tan_2phi, cos_phi, sin_phi, ZZ, ptbl, rank_T, freezed_rank_T)
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        #print('[U_Function backward]')
        tan_2phi, cos_phi, sin_phi, ZZ, ptbl, rank_T, freezed_rank_T = ctx.saved_tensors
        BS = int(grad_Y.shape[0])
        N = int(grad_Y.shape[1])
        grad_Z2 = torch.empty((BS, N), dtype=grad_Y.dtype, device=grad_Y.device, requires_grad=False)
        grad_tan_2phi = torch.empty((BS, int(N*(N-1)/2)),
                                    dtype=grad_Y.dtype, device=grad_Y.device, requires_grad=False)
        sow_cpp.backward_Uh(grad_Y, ptbl, rank_T, freezed_rank_T, tan_2phi,
                            cos_phi, sin_phi, ZZ, grad_Z2, grad_tan_2phi)
        # 全サンプル-mean方式
        #grad_tan_2phi = grad_tan_2phi.mean(axis=0).view(1,-1)

        '''
        len_in = float(torch.sum(torch.diag(grad_Y.mm(grad_Y.t()))))
        len_out = float(torch.sum(torch.diag(grad_Z2.mm(grad_Z2.t()))))
        if abs((len_in - len_out) / len_in) > -1.0e-15 :
            print(f'BS={BS}, N={N}')
            print('ptbl:', ptbl.shape, ptbl.dtype, ptbl.device, ptbl.requires_grad)
            print_ptbl(ptbl)
            print('tan_2phi:', tan_2phi.shape, tan_2phi.dtype, tan_2phi.device, tan_2phi.requires_grad)
            print(tan_2phi)
            print('grad_Y:', grad_Y.shape, grad_Y.dtype, grad_Y.device, grad_Y.requires_grad)
            print(grad_Y)
            print('grad_Z2:', grad_Z2.shape, grad_Z2.dtype, grad_Z2.device, grad_Z2.requires_grad)
            print(grad_Z2)
            #print(torch.diag(grad_Y.mm(grad_Y.t())))
            #print(torch.diag(grad_Z2.mm(grad_Z2.t())))
            print(f'abs((|grad_Y|-|grad_Z2|)/|grad_Y|) = (|%.3e| - |%.3e|) / |%.3e| = %.3e' %
                  (len_in, len_out, len_in, abs((len_in - len_out) / len_in)))
            print('grad_tan_2phi:')
            print(grad_tan_2phi)
            #exit()
        '''

        #print(' ctx.needs_input_grad:',  ctx.needs_input_grad)
        if ctx.needs_input_grad[0] == False : grad_Z2 = None
        if ctx.needs_input_grad[1] == False : grad_tan_2phi = None
        # ptbl, rid, ttbl has no grad.
        return grad_Z2, grad_tan_2phi, None, None, None, None, None

class S_Function(Function):
    """Output & Gradient Calcuation for Z2=Z1*Sigma """
    @staticmethod
    def forward(ctx, Z1, a, S_max, rank):
        #print('[S_Function forward]')
        BS = int(Z1.shape[0])
        N = int(Z1.shape[1])
        rank_T = torch.tensor(rank, dtype=torch.int, device=Z1.device, requires_grad=False)
        #
        #f = 1.0 / (1.0 + torch.exp(-a))
        #S = torch.zeros([N], dtype=a.dtype, device=a.device, requires_grad=False)
        #S[0] = S_max * f[0]
        #for i in range(1, rank): S[i] = S[i-1] * f[i]
        #Z2 = Z1 * S
        #
        f = torch.empty(N, dtype=Z1.dtype, device=Z1.device, requires_grad=False)
        S = torch.zeros(N, dtype=Z1.dtype, device=Z1.device, requires_grad=False)
        Z2 = torch.empty((BS, N), dtype=Z1.dtype, device=Z1.device, requires_grad=False)
        sow_cpp.forward_S(Z1, S_max, rank_T, a, f, S, Z2)

        '''
        print(f'BS={BS}, N={N}, R={rank}')
        print('a', a.shape, a.dtype, a.device)
        print(a)
        print('f', f.shape, f.dtype, f.device, f.requires_grad)
        print(f)
        print('S', S.shape, S.dtype, S.device)
        print(S)
        print('Z1', Z1.shape, Z1.dtype, Z1.device)
        print(Z1)
        print('Z2', Z2.shape, Z2.dtype, Z2.device)
        print(Z2)
        '''

        ctx.save_for_backward(Z1, rank_T, f, S, Z2)
        return Z2

    @staticmethod
    def backward(ctx, grad_Z2):
        #print('[S_Function backward]')
        Z1, rank_T, f, S, Z2 = ctx.saved_tensors
        BS = grad_Z2.shape[0]
        N = grad_Z2.shape[1]
        
        '''
        F = torch.tril(torch.ones(N, N, dtype=grad_Z2.dtype, device=grad_Z2.device) - f)
        print('F:', F.shape, F.dtype, F.device, F.requires_grad)
        print(F)
        grad_Z1 = grad_Z2 * S
        grad_a = (grad_Z2 * Z2).mm(F)
        tmp = grad_Z2 * Z2
        print('tmp:', tmp.shape, tmp.dtype, tmp.device, tmp.requires_grad)
        print(tmp)
        '''

        #'''
        grad_Z1 = torch.empty((BS, N), dtype=grad_Z2.dtype, device=grad_Z2.device)
        grad_a =  torch.empty((BS, N), dtype=grad_Z2.dtype, device=grad_Z2.device)
        sow_cpp.backward_S(grad_Z2, rank_T, Z2, S, f, grad_Z1, grad_a)
        #'''

        # 全サンプル-mean方式（→性能劣化を招くので全サンプル使用の方がよさそう → 次行をコメントアウト）
        #grad_a = grad_a.mean(axis=0).view(1,-1)

        '''
        print('Z1', Z1.shape, Z1.dtype, Z1.device)
        print(Z1)
        print('S', S.shape, S.dtype, S.device)
        print(S)
        print('Z2:', Z2.shape)
        print(Z2)
        print('grad_Z2', grad_Z2.shape, grad_Z2.dtype, S.device)
        print(grad_Z2)
        print('grad_Z1:', grad_Z1.shape, grad_Z1.dtype, grad_Z1.device)
        print(grad_Z1)
        print('grad_a:', grad_a.shape, grad_a.dtype, grad_a.device)
        print(grad_a)
        '''

        if torch.any(torch.isnan(grad_a)) : exit()
        return grad_Z1, grad_a, None, None

class SOW_V3(nn.Module):
    def __init__(self, M, init_file, num_tasks=1, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.reset_parameters(M, init_file, num_tasks, device, dtype)
        #
        self.U_func = U_Function.apply
        self.S_func = S_Function.apply
        self.V_func = V_Function.apply
        # for debug
        #self.init_tan_2phi = torch.clone(self.tan_2phi).detach()
        #self.init_a = torch.clone(self.a).detach()
        #self.init_tan_2theta = torch.clone(self.tan_2theta).detach()
        return

    def reset_parameters(self, M, init_file, num_tasks, device, dtype):
        """各パラメータの初期値を設定"""
        self.device = device
        self.dtype = dtype
        N = 2 ** M
        self.N = N
        self.max_rank = N
        self.rank = self.max_rank	# 学習時や推論時に使用するランク
        self.freezed_rank = 0		# For Continual Learning
        self.num_tasks = num_tasks	# For Continual Learning

        ## 初期値データの読み込み
        print(f'{init_file}: loading..')
        sow_dict = torch.load(init_file)
        # 定数 ptbl, ranks, ttbl, S_max の設定
        self.ptbl = sow_dict['ptbl'].to(device=device)
        #self.ranks = sow_dict['ranks'].to(device=device)
        #self.ttbl = sow_dict['ttbl'].to(device=device)
        self.S_max = sow_dict['S_max'].to(device=device)
        # パラメータ tan_2phi, a, tan_2theta の設定（phi, theta は -pi/4 [rad] 以上, +pi/4 未満）
        self.tan_2phi = nn.Parameter(sow_dict['tan_2phi'].to(device=device))
        self.a = nn.Parameter(sow_dict['a'].to(device=device))
        self.tan_2theta = nn.Parameter(sow_dict['tan_2theta'].to(device=device))
        
        '''
        # 読み込み結果の確認
        print('  S_max:', self.S_max)
        print('  ptbl:', self.ptbl.shape, self.ptbl.dtype, self.ptbl.device)
        print_ptbl(self.ptbl)
        print_pidx(self.ptbl)
        print('  ranks:', self.ranks)
        print('  ttbl:', self.ttbl.shape, self.ttbl.dtype, self.ttbl.device)
        print('  a:', self.a.shape, self.a.dtype, self.a.device)
        exit()
        '''
        
        # ワーク領域
        self.cos_phi = torch.empty((1, int(N*(N-1)/2)), dtype=dtype, device=device, requires_grad=False)
        self.sin_phi = torch.empty((1, int(N*(N-1)/2)), dtype=dtype, device=device, requires_grad=False)
        self.cos_theta = torch.empty((1, int(N*(N-1)/2)), dtype=dtype, device=device, requires_grad=False)
        self.sin_theta = torch.empty((1, int(N*(N-1)/2)), dtype=dtype, device=device, requires_grad=False)
        self.S = torch.zeros(N, dtype=dtype, device=device, requires_grad=False)
        #self.ZZ_U = torch.empty((1, N*N*BS), dtype=dtype, device=device, requires_grad=False)
        #self.ZZ_V = torch.empty((1, N*N*BS), dtype=dtype, device=device, requires_grad=False)

	# For saving best low-rank a at each specified task
        self.low_rank_a = torch.full((num_tasks, N), fill_value=torch.finfo(dtype).min,
                                     dtype=dtype, device=device, requires_grad=False)
	# For saving rank at each specified task
        self.ranks = torch.empty((self.num_tasks), dtype=torch.int, device=device, requires_grad=False)
        # For backup of a
        self.a_backup = None

        # 推論用 W (= Ｕ×Σ×Ｖ^T)
        self.W = torch.empty((N, N), dtype=dtype, device=device, requires_grad=False)

    # V の収束確認を行うために使用（rankを使用）
    def forward_V(self, X):
        Z1 = self.V_func(X, self.tan_2theta, self.ptbl, self.rank, self.freezed_rank)
        return Z1

    # S の収束確認を行うために使用（rankを使用，Sは部分フリーズを適用しない）
    def forward_S(self, Z1):
        Z2 = self.S_func(Z1, self.a, self.S_max, self.rank)
        return Z2

    # U^T の収束確認を行うために使用（rankを使用）
    def forward_U(self, Z2):
        Y = self.U_func(Z2, self.tan_2phi, self.ptbl, self.rank, self.freezed_rank)
        return Y

    # Forward
    def forward(self, X):
        Z1 = self.V_func(X, self.tan_2theta, self.ptbl, self.rank, self.freezed_rank)
        Z2 = self.S_func(Z1, self.a, self.S_max, self.rank)
        Y = self.U_func(Z2, self.tan_2phi, self.ptbl, self.rank, self.freezed_rank)
        return Y

    def extra_repr(self):
        return 'in_features={}, out_features={}, num_tasks={}, dtype={}, device={}'.format(
            self.N, self.N, self.num_tasks, self.dtype, self.device
        )

    def update_freezed_rank(self, r):
        if self.freezed_rank < r : self.freezed_rank = r
        return

    def get_freezed_rank(self):
        return self.freezed_rank
    
    def freeze_by_r(self, r):
        # a を指定数まで低ランク化
        tmp = torch.clone(self.a).detach()
        tmp[r:] = torch.finfo(tmp.dtype).min
        self.a = nn.Parameter(tmp)
        return

    def backup_full_rank_a(self):
        self.a_backup = torch.clone(self.a).detach()
        return

    def restore_full_rank_a(self):
        self.a = torch.nn.Parameter(self.a_backup)
        return

    def set_low_rank_a(self, rank):
        tmp = torch.clone(self.a_backup).detach()
        tmp[rank:] = torch.finfo(tmp.dtype).min
        self.a = nn.Parameter(tmp)
        return

    def save_low_rank_a(self, task_id):
        self.low_rank_a[task_id] = torch.clone(self.a).detach()
        return

    def load_low_rank_a(self, task_id):
        self.a = nn.Parameter(torch.clone(self.low_rank_a[task_id]).detach())
        return

    def save_rank(self, task_id, rank):
        self.ranks[task_id] = rank
        return
    
    def get_rank(self, task_id):
        return self.ranks[task_id]

    def set_W(self, rank):
        U = torch.empty((self.N, self.N), dtype=self.dtype, device=self.device, requires_grad=False)
        S = torch.empty((self.N), dtype=self.dtype, device=self.device, requires_grad=False)
        Vh = torch.empty((self.N, self.N), dtype=self.dtype, device=self.device, requires_grad=False)
        #sow_cpp.setup_U(ptbl, sow.tan_2phi, U)
        #sow_cpp.setup_S(self.a, S)
        #sow_cpp.setup_Vh(ptbl, sow.tan_2theta, Vh)
        if rank < self.max_rank : S[r:] = 0.0
        self.W = (U * S).mm(Vh)
