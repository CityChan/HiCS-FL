import os
import os.path
import sys
import logging
import copy
import time
import torch
from server.server_base import Server_base
from server.server_poc import Server_PowOfChoice
from server.server_divfl import Server_DivFL
from server.server_cs import Server_CS
from server.server_hics import Server_HiCS
from server.server_fedcor import Server_FedCor


def train(args):
    seed = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    device = device.split(',')
    print(args)
    args['seed'] = seed
    args['device'] = device
    _train(args)

    myseed = 42  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        
        
        
        
def _train(args):
    if args["method"] == "random":
        server = Server_base(args)
    if args["method"] == "poc":
        server = Server_PowOfChoice(args)
    if args["method"] == "divfl":
        server = Server_DivFL(args)
    if args["method"] == "cs":
        server = Server_CS(args)
    if args["method"] == "hics":
        server = Server_HiCS(args)
    if args["method"] == "fedcor":
        server = Server_FedCor(args)
    server.train()
    
