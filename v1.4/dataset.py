### Main module to preprocess dataset

import pandas as pd
from tqdm import tqdm
import pickle as pkl
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import CHARAS_LIST
import joblib
import statistics as stat
import preprocess as p
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def calSharpeRatio(months, rfdf, df, charac = "predictions", col = "predictions"):
    """
    Calculate Sharpe Ratio of the portfolio
    """
    rfdf = rfdf.set_index("month")
    monthly_means = []
    monthly_portfolio = []
    for d in months:
        rf = rfdf.loc[d].RF
        portfolio_ret = cal_portfolio_ret(it = (d, charac), df = df, col=col)
        monthly_means.append(portfolio_ret - rf)
        monthly_portfolio.append(portfolio_ret)
    
    mu = stat.mean(monthly_means)
    sd = stat.stdev(monthly_portfolio)
    return mu/sd, monthly_means


def cal_portfolio_ret(it, df, col = "ret-rf"):
    d, f = it[0], it[1]
    long_portfolio = df.loc[df.month == d][['permno', f]].sort_values(by=f, ascending = False)[:df.loc[df.month == d].shape[0]//10]['permno'].to_list()
    short_portfolio = df.loc[df.month == d][['permno', f]].sort_values(by=f, ascending = False)[-df.loc[df.month == d].shape[0]//10:]['permno'].to_list()
    long_ret = df.loc[df.month == d].drop_duplicates('permno').set_index('permno').reindex(long_portfolio)[col].dropna().mean()
    short_ret = df.loc[df.month == d].drop_duplicates('permno').set_index('permno').reindex(short_portfolio)[col].dropna().mean()
    chara_ret = 0.5 * (long_ret - short_ret)
    return chara_ret


def transform(data, name,  batch_size = 512):
    d = RData(data=data)
    loader = torch.utils.data.DataLoader(d, batch_size = batch_size)
    torch.save(loader, name)
    return loader


class RData(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index], self.data[2][index]
    
    def __len__(self):
        return self.data[0].shape[0]



class RetData:
    def __init__(self, datapath = "../datasets/dataset_final.pkl"):
        self.datapath = datapath
        self.load_data(datapath)
        print(self.df.columns)
        self.scaler = StandardScaler()
        self.scale_data()
    
    def load_data(self, datapath):
        print("-------In Load Data-------")
        with open(datapath, "rb") as fp:
            self.df = pkl.load(fp)

    def scale_data(self):
        # save the scaler object for return_7
        ret_scaler_single = StandardScaler()
        ret_scaler_single.fit_transform(self.df[['ret-rf_6']])
        # joblib.dump(ret_scaler_single, "./output/ret_scaler_single.obj")
        joblib.dump(ret_scaler_single, "./output/ret_scaler_single_2016.obj")
        del ret_scaler_single

        # standardize charas
        print("Standardizing all features...")
        chara_cols = []
        for i in range(0,7):
            for chara in CHARAS_LIST:
                chara_cols.append(f"{chara}_{i}")

        chara_scaler_all = StandardScaler()
        chara_scaler_all.fit_transform(self.df[chara_cols])
        # joblib.dump(chara_scaler_all, "./output/chara_scaler_all.obj")
        joblib.dump(chara_scaler_all, "./output/chara_scaler_all_2016.obj")
        del chara_scaler_all

        # standardize all
        print("Standardizing all features...")
        self.cols = [f"ret-rf_{i}" for i in range(0,7)] + chara_cols
        self.df[self.cols] = self.scaler.fit_transform(self.df[self.cols])
        # joblib.dump(self.scaler, "./output/scaler_all_features.obj")
        joblib.dump(self.scaler, "./output/scaler_all_features_2016.obj")
        del self.scaler, self.cols


    def _makedata(self, df):
        dates = list(df["month"].unique())
        # permnos = list(df.permno.unique())

        data = {}

        for dt in tqdm(dates):
            df_date = df[df["month"] == dt].sort_values('permno')
            xs = []     # encoder inputs (r1-r4, z1-z4)
            xds = []    # decoder inputs (r5-r6, z5-z6)
            ys = []     # returns (r6-r7)

            ############### xs ################
            charas = [] 
            for i in range(0,4):
                charas.append(list(df_date[f"ret-rf_{i}"]))

                for chara in CHARAS_LIST:
                    charas.append(list(df_date[f"{chara}_{i}"]))

            xs.append(charas)

            ############## xds #################

            for i in range(4,6):
                xds.append(list(df_date[f"ret-rf_{i}"]))

                for chara in CHARAS_LIST:
                    xds.append(list(df_date[f"{chara}_{i}"]))

            ############### ys ################
            for i in range(5,7):
                ys.append(list(df_date[f"ret-rf_{i}"]))
            

            data[dt] = {
                "xs" : torch.FloatTensor(xs),
                "xds" : torch.FloatTensor(xds),
                "ys" : torch.FloatTensor(ys)
            }

        return data
            

    def loaders(self):
        print("-------In Loaders-------")

        # df_train = self.df[(self.df['month'] >= 195700) & (self.df['month'] <= 197412)]
        # df_valid = self.df[(self.df['month'] >= 197500) & (self.df['month'] <= 198612)]
        # df_test = self.df[self.df['month'] >= 198700]

        df_train = self.df[(self.df['month'] >= 201601) & (self.df['month'] <= 201602)]
        df_valid = self.df[(self.df['month'] >= 201603) & (self.df['month'] <= 201604)]
        df_test = self.df[self.df['month'] >= 201605]

        print("pickle saving...")
        # df_test.to_pickle("/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_test.pkl")
        # df_train.to_pickle("/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_train.pkl")
        # df_valid.to_pickle("/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_valid.pkl")

        df_test.to_pickle("/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_test_2016.pkl")
        df_train.to_pickle("/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_train_2016.pkl")
        df_valid.to_pickle("/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_valid_2016.pkl")

        self.x_train = self._makedata(df_train)
        self.x_valid = self._makedata(df_valid)
        self.x_test = self._makedata(df_test)

        del df_test , df_train, df_valid
        
        #############################################

        print("-------Making tensors--------")

        train = self.loadTensors(self.x_train)
        valid = self.loadTensors(self.x_valid)
        test = self.loadTensors(self.x_test)

        print("tensor saving...")
        # torch.save(test, "/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_test.pt")
        # torch.save(train, "/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_train.pt")
        # torch.save(valid, "/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_valid.pt")

        torch.save(test, "/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_test_2016.pt")
        torch.save(train, "/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_train_2016.pt")
        torch.save(valid, "/data/sgupta91/TimeSeriesTransformer/v1.4/output/Pred_valid_2016.pt")

        del train, valid, test


    def loadTensors(self, X:dict):
        encoder_input = []
        decoder_input = []
        targets = []
        for key, _ in X.items():
            encoder_input.append(X[key]['xs'].squeeze(dim = 0))
            decoder_input.append(X[key]["xds"])
            targets.append(X[key]["ys"])

        print(encoder_input[1].shape)
        print(decoder_input[1].shape)
        print(targets[1].shape)

        encoder_input = torch.cat(encoder_input, dim = 1)
        decoder_input = torch.cat(decoder_input, dim = 1)
        targets = torch.cat(targets, dim = 1)

        return torch.transpose(encoder_input, 0, 1), torch.transpose(decoder_input, 0, 1), torch.transpose(targets, 0,1)














    

    




        




        