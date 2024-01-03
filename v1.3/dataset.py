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
    def __init__(self, returnpath = "../datasets/month_ret.pkl", datapath = "../datasets/dataset.csv"):

        self.retpath = returnpath
        self.datapath = datapath
        self.scaler = StandardScaler()

        self.loadData()
        self.dates = self.getDates()
        self.rankNormalize()
        self.modify()
        self.joinRets()
        

    def getDates(self):
        dates = sorted(list(self.ret.month.unique()))
        return dates

    def loadData(self):
        self.df = pd.read_csv(self.datapath)
        with open(self.retpath, "rb") as fp:
            self.ret = pkl.load(fp)

    def joinRets(self):
        print("----------In joinrets------------")

        self.ret.rename(columns = {"month" : "month_0"}, inplace = True)

        self.ret = pd.merge(self.ret, self.df, how = "inner", left_on = ["permno", f"month_{3}"], right_on = ["permno", "month"])
        self.ret.drop("month", axis = 1, inplace = True)

        for chara in CHARAS_LIST:
            self.ret.rename(columns = {chara : f"{chara}_{3}"}, inplace = True)

        # for i in tqdm(range(3,5)):
        #     self.ret = pd.merge(self.ret, self.df, how = "inner", left_on = ["permno", f"month_{i}"], right_on = ["permno", "month"])
            
        #     for chara in CHARAS_LIST:
        #         self.ret.rename(columns = {chara : f"{chara}_{i}"})


        # self.ret = pd.merge(self.ret, self.df )

        # for i in tqdm(range(0, 7)):
        #     if i == 0:
        #         self.ret = pd.merge(self.ret, self.df, how= "inner", on = ['permno', 'month'])
        #         self.ret.rename(columns = {"month" : "month_0"}, inplace = True)
        #     else:
        #         self.ret = pd.merge(self.ret, self.df, left_on = ["permno", f"month_{i}"], right_on=["permno", "month"], how = "inner")
        #         self.ret.drop("month", inplace = True)
        #     for chara in CHARAS_LIST:
        #         self.ret.rename(columns = {chara : f"{chara}_{i}"})

        #     print("month" in list(self.ret.columns))

    def rankNormalize(self):
        print("----------In rank normalize-----------")
        dfs = []

        months = list(self.df.DATE)
        months = [int(str(m)[:6]) for m in months]
        self.df['month'] = months

        self.df = self.df.drop(['DATE'], axis = 1)

        months = self.df.month.unique()

        for date in tqdm(months):
            cross_slice = self.df.loc[self.df.month == date].copy(deep = False)
            omitted_mask = 1.0 * np.isnan(cross_slice.loc[cross_slice['month'] == date])
            cross_slice.loc[cross_slice['month'] == date] = cross_slice.fillna(0) + omitted_mask * cross_slice.median()
            cross_slice.loc[cross_slice['month'] == date] = cross_slice.fillna(0)
            re_df = []
            for col in CHARAS_LIST:
                series = cross_slice[col]
                de_duplicate_slice = pd.DataFrame(series.drop_duplicates().to_list(), columns=['chara'])
                series = pd.DataFrame(series.to_list(), columns = ['chara'])
                de_duplicate_slice['sort_rank'] = de_duplicate_slice['chara'].argsort().argsort()
                rank = pd.merge(series, de_duplicate_slice, left_on='chara', right_on='chara', how='right')['sort_rank']
                rank_normal = ((rank - rank.min())/(rank.max() - rank.min())*2 - 1)
                re_df.append(rank_normal)
            re_df = pd.DataFrame(re_df, index = CHARAS_LIST).T.fillna(0)
            re_df['permno'] = list(cross_slice['permno'].astype(int))
            re_df['month'] = list(cross_slice['month'].astype(int))
            dfs.append(re_df)
        self.df = pd.concat(dfs)

    def modify(self):
        print("--------In Modify---------")
        temp_ret = self.ret
        temp_ret = temp_ret.drop('date', axis = 1)
        self.dates = sorted(self.dates)
        df_month = pd.DataFrame({'month' : self.dates, 'month_1' : self.dates, 'month_2' : self.dates, 
                                 'month_3' : self.dates, 'month_4' : self.dates, 'month_5' : self.dates, 'month_6' : self.dates })
        df_month['month_1'] = df_month["month_1"].shift(-1)
        df_month['month_2'] = df_month["month_2"].shift(-2)
        df_month["month_3"] = df_month["month_3"].shift(-3)
        df_month["month_4"] = df_month["month_4"].shift(-4)
        df_month["month_5"] = df_month["month_5"].shift(-5)
        df_month["month_6"] = df_month["month_6"].shift(-6)
        df_month.dropna(inplace=True)

        self.ret = pd.merge(self.ret, df_month, on="month", how="inner")
        
        self.ret = self.ret.drop("ret-rf", axis = 1)
        self.ret = self.ret.drop("date", axis = 1)

        for i in range(0,7):
            if i == 0:
                self.ret = pd.merge(self.ret, temp_ret, left_on = ["permno", "month"], right_on = ["permno", "month"])
                self.ret.rename(columns={"ret-rf" : f"ret-rf_{i}"}, inplace=True )
            else:
                self.ret = pd.merge(self.ret, temp_ret, left_on=["permno", f"month_{i}"], right_on=["permno", "month"])
                self.ret.rename(columns={"ret-rf" : f"ret-rf_{i}", "month_x" : "month"}, inplace=True )
                self.ret = self.ret.drop("month_y", axis = 1)

            self.ret[[f"ret-rf_{i}"]] = StandardScaler().fit_transform(self.ret[[f"ret-rf_{i}"]])
    
    def _makedata(self, df):
        dates = list(df["month_0"].unique())
        permnos = list(df.permno.unique())

        data = {}

        for dt in tqdm(dates):
            df_date = df[df["month_0"] == dt].sort_values('permno')
            xs = [] 
            xds = []
            ys = []

            ###############################
            charas = []
            for chara in CHARAS_LIST:
                charas.append(list(df_date[f"{chara}_{3}"]))

            for i in range(0,4):
                charas.append(list(df_date[f"ret-rf_{i}"]))

            xs.append(charas)

            ###############################

            for i in range(4,6):
                xds.append(list(df_date[f"ret-rf_{i}"]))


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

        df_train = self.ret[(self.ret['month_0'] >= 195700) & (self.ret['month_0'] <= 197412)]
        df_valid = self.ret[(self.ret['month_0'] >= 197500) & (self.ret['month_0'] <= 198612)]
        df_test = self.ret[self.ret['month_0'] >= 198700]

        self.x_train = self._makedata(df_train)
        self.x_valid = self._makedata(df_valid)
        self.x_test = self._makedata(df_test)
        
        print("-------Making tensors--------")

        train = self.loadTensors(self.x_train)
        valid = self.loadTensors(self.x_valid)
        test = self.loadTensors(self.x_test)

        return train, valid, test


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














    

    




        




        