import torch.nn as nn
import torch.optim as optim
from progress.bar import Bar
import time
import torch
import copy
from torch.nn import functional as F
from tqdm import tqdm
import os
import numpy as np
from utils import Accuracy

class Client_base(object):
    def __init__(self, args,idx,model,trainloader, testloader):
        self.args = args
        self.idx = idx
        self.device = args["device"][0]
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        if args["optim"] == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args["lr"],momentum=self.args["momentum"],weight_decay=self.args["weight_decay"])
        elif args["optim"] == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
    def _update_prox(self,prev_model):
        batch_loss = []
        ce_loss = nn.CrossEntropyLoss() 
        global_weight_collector = list(prev_model.parameters())
        for batch_idx, (X, y) in enumerate(self.trainloader):
            X = X.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            p = self.model(X).double()
            loss1 = ce_loss(p,y) 
            fed_prox_reg = 0.0
            mu = 0.5
            for param_index, param in enumerate(self.model.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss = loss1 + fed_prox_reg
            loss.backward()
            batch_loss.append(loss1.item())
            self.optimizer.step()
        return batch_loss

    def _update(self):
        batch_loss = []
        ce_loss = nn.CrossEntropyLoss() 
        for batch_idx, (X, y) in enumerate(self.trainloader):
            X = X.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            p = self.model(X).double()
            loss = ce_loss(p,y) 
            loss.backward()
            batch_loss.append(loss.item())
            self.optimizer.step()
        return batch_loss
        

    def local_training(self):
        self.model.train()
        prev_model = copy.deepcopy(self.model)
        prev_model.eval()
        for epoch in range(self.args["local_epochs"]):
            if self.args["loss"] == "prox":
                batch_loss = self._update_prox(prev_model)
            if self.args["loss"] == "avg":
                batch_loss = self._update()
 
        return self.model.state_dict(), np.sum(np.array(batch_loss))/len(batch_loss)

    def local_accuracy(self):
        self.model.eval()
        accuracy = 0
        cnt = 0
        for batch_idx, (X, y) in enumerate(self.testloader):
            X = X.to(self.device)
            y = y.to(self.device)
            p = self.model(X).double()
            y_pred = p.argmax(1)
            accuracy += Accuracy(y,y_pred)
            cnt += 1
        return accuracy/cnt

    def train_loss(self):
        self.model.eval()
        ce_loss = nn.CrossEntropyLoss() 
        batch_loss = []
        for batch_idx, (X, y) in enumerate(self.trainloader):
            X = X.to(self.device)
            y = y.to(self.device)
            p = self.model(X).double()
            loss = ce_loss(p,y)  
            batch_loss.append(loss.item())
            
        return np.sum(np.array(batch_loss))/len(batch_loss)

    def load_model(self,global_weights):
        self.model.load_state_dict(global_weights)