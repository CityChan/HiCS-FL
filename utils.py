import torch
import copy
import numpy as np
from itertools import product
from scipy.cluster.hierarchy import fcluster
from copy import deepcopy
from torch import nn
import torch.optim as optim

def Accuracy(y,y_predict):
    leng = len(y)
    miss = 0
    for i in range(leng):
        if not y[i]==y_predict[i]:
            miss +=1
    return (leng-miss)/leng


def average_weights(w):
    """
    average the weights from all local models
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


        
def get_gradients(sampling, global_m, local_models):
    """return the `representative gradient` formed by the difference between
    the local work and the sent global model"""

    local_model_params = []
    for model in local_models:
        local_model_params += [
            [tens.detach().cpu().numpy() for tens in list(model.parameters())]
        ]

    global_model_params = [
        tens.detach().cpu().numpy() for tens in list(global_m.parameters())
    ]

    local_model_grads = []
    for local_params in local_model_params:
        local_model_grads += [
            [
                local_weights - global_weights
                for local_weights, global_weights in zip(
                    local_params, global_model_params
                )
            ]
        ]

    return local_model_grads


def get_gradients_fc(sampling, global_m, local_models):
    """return the `representative gradient` formed by the difference between
    the local work and the sent global model"""

    local_model_params = []
    for model in local_models:
        local_model_params +=  [
           [tens.detach().cpu().numpy() for tens in list(model.parameters())[-2:]]
        ]
            
    global_model_params = [
        tens.detach().cpu().numpy() for tens in list(global_m.parameters())[-2:]
    ]
    
    
    local_model_grads = []
    for local_params in local_model_params:
        local_model_grads += [
            [
                local_weights - global_weights
                for local_weights, global_weights in zip(
                    local_params, global_model_params
                )
            ]
        ]
    return local_model_grads




