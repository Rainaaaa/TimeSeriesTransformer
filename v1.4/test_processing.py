from dataset import RetData, transform
from modules2 import FinTransformer
from train import trainer, tester
import torch
from utils import CHARAS_LIST
from tqdm import tqdm
from logger import Logger
import sys
import os

import datetime

import pandas as pd
import pickle as pkl

import warnings
warnings.filterwarnings('ignore')

# 1. Load data
# char_data = pd.read_csv("../datasets/dataset.csv")
# with open('../datasets/month_ret.pkl', "rb") as fp:
#     month_ret = pkl.load(fp)


def saveObj(obj, filepath = './output/r.pkl' ):
    print("saving processed data obj...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as fp:
        r = pkl.dumps(obj)
        fp.write(r)

def readLoad(filepath = './output/r.pkl'):
    print("loading processed data obj...")
    with open(filepath, "rb") as fp:
        r = pkl.loads(fp.read())
    return r

### import small data
r = RetData(datapath = "../datasets/dataset_final.pkl")

### save processed data ({company_num}_{chara_num}_{year_range})
saveObj(r)


# ### process data
# train, valid, test, df_train, df_valid, df_test = r.loaders()
# print("-------Data loaded-------")

# print("tensor saving...")
# torch.save(test, "/data/sgupta91/TimeSeriesTransformer/v1.3/output/Pred_test.pt")
# torch.save(train, "/data/sgupta91/TimeSeriesTransformer/v1.3/output/Pred_train.pt")
# torch.save(valid, "/data/sgupta91/TimeSeriesTransformer/v1.3/output/Pred_valid.pt")

# print("pickle saving...")
# df_test.to_pickle("/data/sgupta91/TimeSeriesTransformer/v1.3/output/Pred_test.pkl")
# df_train.to_pickle("/data/sgupta91/TimeSeriesTransformer/v1.3/output/Pred_train.pkl")
# df_valid.to_pickle("/data/sgupta91/TimeSeriesTransformer/v1.3/output/Pred_valid.pkl")

# # 2. setup training 
# trainloader = transform(data = train, name = "trainloader.pt", batch_size =2048 )
# testloader = transform(data = test, name = "testloader.pt", batch_size = 2048)
# validloader = transform(data = valid, name = "validloader.pt", batch_size = 2048)

# model = FinTransformer(embed_size=512, 
#                        inner_dim = 512, 
#                        num_returns=8,     #98 (company num)
#                        num_decoder_returns=2, 
#                        heads=8, 
#                        repeats=1, dropout=0.1)

# device = "cuda"
# model = model.to(device)


# # 3. train# Parameter setting
# timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
# filename = f'log_{timestamp}.txt'
# epoch = 1000
# lr = 1e-9   #1e-6 to 1e-9
# batch_size = 2048

# with Logger(filename):

#     print(f'================= Epochs: {epoch}; lr: {lr}; batch_size: {batch_size} =================')

#     # training
#     trainer(model=model, Epochs=epoch, lr = lr, trainloader=trainloader, validloader=validloader, label = timestamp)

# # 4. save the result
# pred_test_res = tester(model = model, testloader=testloader,label = timestamp)