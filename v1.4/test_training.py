from dataset import RetData, transform
from modules2 import FinTransformer
from train import trainer, tester
import torch
from utils import CHARAS_LIST
from tqdm import tqdm
from logger import Logger
import sys

import datetime

import pandas as pd
import pickle as pkl

import warnings
warnings.filterwarnings('ignore')

import argparse
import os

def readLoad(filepath = './output/r.pkl'):
    print("loading processed data obj...")
    with open(filepath, "rb") as fp:
        r = pkl.loads(fp.read())
    return r

def parse_args():
    parser = argparse.ArgumentParser(description="Slurm Job Parser")

    parser.add_argument("--job_name", type=str, help="Job name", required=True)
    parser.add_argument("--company_num", type=int, help="Number of companies", required=False)
    parser.add_argument("--chara_num", type=int, help="Number of characteristics", required=False)
    parser.add_argument("--year_range", type=str, help="Range of years", required=False)
    parser.add_argument("--processed_obj", type=str, help="Processed object path", required=True)
    parser.add_argument("--checkpoint", type=str, help="Trained model checkpoint", required=False)

    parser.add_argument("--epoch", type=int, help="Number of epochs", required=True)
    parser.add_argument("--lr", type=str, help="Learning rate", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size", required=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    year_range = 'all' if not args.year_range else args.year_range
    job_name = args.job_name
    company_num = 'all' if not args.company_num else args.company_num
    chara_num = 'all' if not args.chara_num else args.chara_num
    obj = args.processed_obj

    
    print(f"------------------{args.job_name}------------------")
    print(f"Numbers of Companies: {company_num}")
    print(f"Numbers of Characteristics: {chara_num}")
    print(f"Year Range: {year_range}")

    # load processed object
    r = readLoad(obj) if obj else readLoad(f"/data/sgupta91/TimeSeriesTransformer/v1.4/output/r.pkl")
   
    # 1. Load data & setup training (these loaders will be saved as the name given (eg."trainloader.pt"))
    # if the loaders did not load before, then load them and save them
    if not os.path.exists(f"/data/sgupta91/TimeSeriesTransformer/v1.4/trainloader_2016.pt"):
        print("data loading...")
        r.loaders()
        # test = torch.load("/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_test.pt")
        # train = torch.load("/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_train.pt")
        # valid = torch.load("/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_valid.pt")

        # 2016
        test = torch.load("/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_test_2016.pt")
        train = torch.load("/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_train_2016.pt")
        valid = torch.load("/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_valid_2016.pt")

        print("making loaders...")
        trainloader = transform(data = train, name = "trainloader.pt", batch_size =2048 )   #2048
        testloader = transform(data = test, name = "testloader.pt", batch_size = 2048)
        validloader = transform(data = valid, name = "validloader.pt", batch_size = 2048)

        del train, test, valid
    else:
        print("loader has been loaded created in advanced; loading...")
        # trainloader = torch.load(f"/data/sgupta91/TimeSeriesTransformer/v1.4/trainloader_all.pt")
        # testloader = torch.load(f"/data/sgupta91/TimeSeriesTransformer/v1.4/testloader_all.pt")
        # validloader = torch.load(f"/data/sgupta91/TimeSeriesTransformer/v1.4/validloader_all.pt")
        
        # 2016
        trainloader = torch.load(f"/data/sgupta91/TimeSeriesTransformer/v1.4/trainloader_2016.pt")
        testloader = torch.load(f"/data/sgupta91/TimeSeriesTransformer/v1.4/testloader_2016.pt")
        validloader = torch.load(f"/data/sgupta91/TimeSeriesTransformer/v1.4/validloader_2016.pt")

    # 2. define model and device
    model = FinTransformer(embed_size=512, 
                        inner_dim = 512, 
                        num_returns=380,     #num of features
                        num_decoder_returns=2, 
                        heads=8, 
                        repeats=1, dropout=0.1)

    device = "cuda"
    model = model.to(device)

    # check if there is input checkpoint for continue training
    if args.checkpoint:
        print("loading checkpoint...")
        model.load_state_dict(torch.load(args.checkpoint))

    # # 3. train# Parameter setting
    timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    filename = f'log_{timestamp}.txt'
    epoch = args.epoch
    lr_str = args.lr.split('-e')
    lr = float(10 ** -(int(lr_str[1])))
    batch_size = args.batch_size 
    print(f"training-{timestamp}...")
    with Logger(filename):

        print(f'================= Epochs: {epoch}; lr: {lr}; batch_size: {batch_size} =================')

        # training
        trainer(model=model, Epochs=epoch, lr = lr, trainloader=trainloader, validloader=validloader, label = timestamp)

    # 4. save the result (tester fxn automatically saves the result) saved in "/data/sgupta91/TimeSeriesTransformer/v1.3/output/Predictions_{label}.csv"
    print(f"saving results...")
    pred_test_res = tester(model = model, testloader=testloader,label = timestamp)