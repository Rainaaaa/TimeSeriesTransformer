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
import argparse

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Slurm Job Parser")

    parser.add_argument("--job_name", type=str, help="Job name")
    parser.add_argument("--company_num", type=int, help="Number of companies")
    parser.add_argument("--chara_num", type=str, help="Number of characteristics")
    parser.add_argument("--test_data", type=str, help="Test dataset path")
    parser.add_argument("--checkpoint", type=str, help="Trained model checkpoint")
    parser.add_argument("--label", type=str, help="Timestamp Label used while training ")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    print(f"------------------{args.job_name}------------------")
    print(f"Numbers of Companies: {args.company_num}")
    print(f"Numbers of Characteristics: {args.chara_num}")


    # 1. Load data
    # test = torch.load("/data/sgupta91/TimeSeriesTransformer/v1.3/output/Pred_test_2016.pt")
    test = torch.load(args.test_data)

    # 2. setup testing loader (these loaders will be saved)
    testloader = transform(data = test, name = "testloader.pt", batch_size = 2048)

    # 3. define model
    model = FinTransformer(embed_size=512, 
                        inner_dim = 512, 
                        num_returns=8,     #98 (company num)
                        num_decoder_returns=2, 
                        heads=8, 
                        repeats=1, dropout=0.1)

    device = "cuda"
    model = model.to(device)


    # 4. tesing Parameter setting
    # timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    # checkpoint = torch.load("/data/sgupta91/TimeSeriesTransformer/v1.3/checkpoint/best_model.pt")
    checkpoint = torch.load(args.checkpoint)

    model.load_state_dict(checkpoint)
    pred_test_res = tester(model = model, testloader=testloader,label = args.label)  # automatically saves the result
