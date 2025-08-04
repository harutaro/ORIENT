#!/usr/bin/env python
# coding: utf-8

import torch
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import copy

import torchvision
import numpy as np
import math

#from args import args as pargs

#
#_______________________________________________________________________________
#
# commented at 20240314_1539
# This code implements the algorithm proposed in the original paper of WSN:
# H. Kang, RJL Mina, et.al.:"Forget-free Continual Learning with Winning
#     Subnetworks," Proc. 39th ICML, 2022.
# They conducted the experiments only for task-incremental continual learning
# to compare WSN with previously proposed methods.
# (See. line3 in section 4).
#
# Since only the task-incremental learning is reported in the original paper,
# this file has no method for conducting the class-incremental learning.
#
# In this implementation,
# we modified the SupSup code previously implemented for FACIL framework.
#_______________________________________________________________________________
#

# Subnetwork forward from hidden networks
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k=0.5):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1-k) * scores.numel())
        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


def get_subnet(scores, k=0.5):
    out = scores.clone()
    _, idx = scores.flatten().sort()
    j = int((1-k) * scores.numel())
    flat_out = out.flatten()
    flat_out[idx[:j]] = 0
    flat_out[idx[j:]] = 1
    return out



class GetSubnetFast(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        return (scores >= 0).float()

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g

def get_subnet_fast(scores, a=0):
    return (scores >= a).float()

def mask_init(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores


def signed_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, 'fan_in')
    gain = nn.init.calculate_gain('relu')
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std


#_______________________________________________________________________________
#
# class Wsn_SubnetLinear(nn.Linear):
#   torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
#     *args = (in_features, out_features)
#     *kwargs = {"bias": True, "device": None, "dtype": None}
#   Wsn_SubnetLinear(*args, *kwargs, no_tasks=1)
# The difference from SupSup_MaskLinear() is that
#   the weights selected in the previous tasks are not trainable.
# This is realized by the set_used_weights_fixed()
#   defined here and inserted in the wsn.train_epoch().
#_______________________________________________________________________________
#
class Wsn_SubnetLinear(nn.Linear):
    def __init__(self, *args, num_tasks=1, device=None, dtype=None, **kwargs):
        super().__init__(*args, device=device, dtype=dtype, **kwargs)
        self.device = device
        self.dtype = dtype
        self.num_tasks = num_tasks
        self.scores = nn.ParameterList(
            [nn.Parameter(mask_init(self)) for _ in range(num_tasks)]
        )
        self.alphas = torch.ones(self.num_tasks, 1, 1) / self.num_tasks
        
        self.weight.requires_grad = True
        signed_constant(self)
        self.task = -1
        if self.bias is not None:
            self.bias.requires_grad = False
        self.num_tasks_learned = 0
        self.cache_masks()
    
    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [get_subnet(self.scores[j].abs()) for j in range(self.num_tasks)]
            )
        )

    def clear_masks(self):
        self.register_buffer("stacked", None)

    def untrainable_mask(self):
        umask = torch.zeros(self.stacked[0].size())
        if self.num_tasks_learned:
            for j in range(self.num_tasks_learned):
                umask = torch.logical_or(umask.to(self.device), self.stacked[j])
        return umask

    def set_used_weights_fixed(self):
        umask = self.untrainable_mask()
        self.weight.grad[umask==1] = 0


    def forward(self, x):
        if self.task < 0: # when the task id is inferred in {oneshot,binary}_inference_method
            # Superimposed forward pass
            alpha_weights = self.alphas[: self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)
            subnet = (
                alpha_weights[idxs].to(self.device) * self.stacked[: self.num_tasks_learned][idxs]
            ).sum(dim=0)
        else:
            # Subnet forward pass (given task info in self.task)
            subnet = GetSubnet.apply(self.scores[self.task].abs())
        
        w = self.weight * subnet
        x = F.linear(x, w, self.bias)
        return x

    def set_task(self, tid):
        self.task = tid
        #print("In SubnetLinear : set task = ", self.task)

    def set_alphas(self, alphas):
        self.alphas = alphas

    def set_num_tasks_learned(self, nt):
        self.num_tasks_learned = nt

    def set_device(self, device):
        self.device = device


    def __repr__(self):
        return f"Wsn_SubnetLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, dtype={self.dtype}, device={self.device})"

# end of class Wsn_SubnetLinear(nn.Linear)


#_______________________________________________________________________________
#
# Utility functions
#_______________________________________________________________________________
#
def set_model_task(model, task, verbose=False):
    for m in model.ListWsn():
        if verbose:
            print(f"=> Set task of {m} to {task}")
        m.task = task

def set_model_device(model, device, verbose=False):
    for m in model.ListWsn():
        if verbose:
            print(f"=> Set device of {m} to {device}")
        m.device = device

def cache_masks(model, verbose=False):
    for m in model.ListWsn():
        if verbose:
            print(f"=> Caching mask state for {m}")
        m.cache_masks()


def set_num_tasks_learned(model, num_tasks_learned, verbose=False):
    for m in model.ListWsn():
        if verbose:
            print(f"=> Setting learned tasks of {m} to {num_tasks_learned}")
        m.num_tasks_learned = num_tasks_learned


def set_alphas(model, alphas, verbose=False):
    for m in model.ListWsn():
        if verbose:
            print(f"=> Setting alphas for {m}")
        m.alphas = alphas

def show_model_requires_grad(model):
    for m in model.ListWsn():
        print("%s.weight.requires_grad = "%(m), m.weight.requires_grad)

'''
def set_used_weights_fixed(model, verbose=False):
    for m in model.ListWsn():
        m.set_used_weights_fixed()
        if verbose:
            print(f"=> Setting used_weights fixed for {m}")
'''
            
#

