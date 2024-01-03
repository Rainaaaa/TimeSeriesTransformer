'''
    Perprocess Module for the datasets for masked transformer and the return_company cross time_series transformer.
    Yt = returns that goes in the Decoder
    Xt = company characateristics that goes in the Encoder

    Future:
    1. Company Mask model - implement the company mask preparation 
'''

'''
    Dict Dataset format : 

    X = {
        date : {
            "ids" : .... 
            "characs" : ....
        }
    }
'''

import pickle as pkl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(path : str):
    '''
        data stored in pkl format
    '''
    with open(path, "rb") as fp:
        data = pkl.load(fp)
    return data

def paddedcharactensors(characs):
    maxl = 18000

    for i in range(len(characs)):
        if characs[i].shape[0] != maxl:
            resl = maxl - characs[i].shape[0]
            rtensor = torch.full((resl, characs[i].shape[1]), -1000, device = device)
            characs[i] = torch.cat([characs[i], rtensor], dim = 0)
    return characs

def paddedIDtensors(ids):
    maxl = 18000

    for i in range(len(ids)):
        if ids[i].shape[0] != maxl:
            # ids[i] = ids[i].unsqueeze(dim=0)
            resl = maxl - ids[i].shape[0]
            rtensor = torch.full((resl, ), -1000, device = device)
            ids[i] = torch.cat((ids[i], rtensor), dim = 0)
    return ids

def padReturns(returns):
    maxl = 18000

    for i in range(len(returns)):
        if returns[i].shape[0] != maxl:
            resl = maxl - returns[i].shape[0]
            rtensor = torch.full((resl,), 0, device = device)
            returns[i] = torch.cat((returns[i], rtensor), dim = 0)
    return returns

def loadTensors_without_Sequential(X:dict):
    characs = []
    ids = []
    returns = []
    prevReturns = []
    for key, _ in X.items():
        characs.append(torch.tensor(X[key]['characs'], device="cuda", dtype=torch.float32))
        ids.append(torch.tensor(X[key]['ids'], device="cuda", dtype=torch.long))
        returns.append(torch.tensor(X[key]['returns'], device= "cuda", dtype = torch.float32))
        prevReturns.append(torch.tensor(X[key]["prev_returns"], device = "cuda", dtype= torch.float32))

    characs = torch.cat(characs, dim = 0)
    ids = torch.cat(ids, dim = 0)
    returns = torch.cat(returns, dim = 0)
    prevReturns = torch.cat(prevReturns, dim = 0)

    return ids, characs, returns, prevReturns

def loadTensors(X : dict):
    characs = []
    ids = []
    returns = []
    
    for key, _ in X.items():
        characs.append(torch.tensor(X[key]['characs'], device="cuda", dtype=torch.float32))
        ids.append(torch.tensor(X[key]['ids'], device="cuda", dtype=torch.long))
        returns.append(torch.tensor(X[key]['returns'], device= "cuda", dtype = torch.float32))
        

    characs = paddedcharactensors(characs)
    ids = paddedIDtensors(ids)
    returns = padReturns(returns)

    characs = torch.stack(characs, dim = 0)
    ids = torch.stack(ids, dim = 0)
    returns = torch.stack(returns, dim = 0)
    # Y = torch.tensor(Y, device = "cuda", dtype=torch.float32)
    return ids, characs, returns


class FData(torch.utils.data.Dataset):
    def __init__(self, ids, characs, returns, prevReturns, datapath = "id2key.pkl"):
        self.ids = ids
        self.chars = characs
        self.returns = returns
        self.prevReturns = prevReturns
        self.path = datapath
        self.id_mask = None
        self.charmask = None
        self.return_mask = None
        self.prevReturns_mask = None
        self.transformIDs()
        self.create_id_mask()
        self.create_charac_mask()
        self.create_return_mask()

    def __len__(self):
        return self.chars.shape[0]

    def __getitem__(self, index):
        return self.ids[index], self.chars[index], self.returns[index], self.prevReturns[index] , self.id_mask[index], self.charmask[index], self.return_mask[index]
    
    def create_id_mask(self):
        self.id_mask = torch.ones_like(self.chars)

    def transformIDs(self):
        id2key = self.load_data(self.path)
        for i in range(self.ids.shape[0]):
            self.ids[i] = id2key[self.ids[i].item()]
        
    def create_id_mask(self):
        self.id_mask = torch.ones_like(self.ids)

    def create_charac_mask(self):
        self.charmask = torch.ones_like(self.chars)
        self.charmask = torch.where(self.chars.isnan(), 0., self.charmask)
        self.chars = torch.where(self.chars.isnan(), 0., self.chars)

    def create_return_mask(self):
        self.return_mask = torch.zeros_like(self.prevReturns)
        

    def load_data(self, path):
        with open(path, "rb") as fp:
            data = pkl.load(fp)
        return data



def transform(datapath, writepath, batch_size):
    with open(datapath, "rb") as fp:
        data = pkl.load(fp)
    
    ids, chars, returns, prevReturns = loadTensors_without_Sequential(data)
    f = FData(ids, chars, returns, prevReturns )
    loader = torch.utils.data.DataLoader(f, batch_size=batch_size)
    torch.save(loader, writepath)

        

    


    


