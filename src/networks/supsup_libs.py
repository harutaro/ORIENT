#!/usr/bin/python
# coding: utf-8

import torch
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import torchvision
import numpy as np
import math

#_______________________________________________________________________________
#
# This code implements an algorithm proposed in the following paper:
# M.Wortsman, et.al.: "Supermasks in Superposition," NeurIPS 2020.
# The main idea of the algorithm is
# adding a supermask as a learning parameter to each layer
# for extracting the subnetwork (the subset of the weights) from the layer.
# Every task has its own subnetwork.
# The other important point is that self.weight.requires_grad = False
# so that self.weight is randomly initialized and fixed.
#_______________________________________________________________________________

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
# class SupSup_MaskLinear(nn.Linear):
#   torch.nn.Linear(*args, device=None, dtype=None, **kwargs)
#     *args = (in_features, out_features)
#     *kwargs = {"bias": True, "device": None, "dtype": None}
#   SupSup_MaskLinear(*args, *kwargs, no_tasks=1)
#_______________________________________________________________________________
#
class SupSup_MaskLinear(nn.Linear):
    def __init__(self, *args, num_tasks=1, device=None, dtype=None, **kwargs):
        super().__init__(*args, device=device, dtype=dtype, **kwargs)
        self.device = device
        self.dtype = dtype
        self.num_tasks = num_tasks
        self.scores = nn.ParameterList(
            [nn.Parameter(mask_init(self)) for _ in range(num_tasks)]
        )
        self.alphas = torch.ones(self.num_tasks, 1, 1) / self.num_tasks
        
        # Keep weights untrained
        self.weight.requires_grad = False
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


    def forward(self, x):
        if self.task < 0: # when the task id is not given
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

    def set_alphas(self, alphas):
        self.alphas = alphas

    '''
    def set_num_tasks_learned(self, nt):
        self.num_tasks_learned = nt
    '''

    '''
    def set_num_tasks_learned(model, num_tasks_learned, verbose=False):
        for m in model.ListSupSup():
            if verbose:
                print(f"=> Setting learned tasks of {m} to {num_tasks_learned}")
            m.num_tasks_learned = num_tasks_learned
    '''
    def set_device(self, device):
        self.device = device


    def __repr__(self):
        return f"SupSup_MaskLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, dtype={self.weight.dtype}, device={self.device})"

# end of class SupSup_MaskLinear(nn.Linear)


#_______________________________________________________________________________
#
# Utility functions
#_______________________________________________________________________________
#
def show_model_requires_grad(model):
    for m in model.ListCustomModules():
        print("%s.weight.requires_grad = "%(m), m.weight.requires_grad)


