    
from modules import FinTransformer
import preprocess as p
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torcheval.metrics import R2Score

device = "cuda" if torch.cuda.is_available() else "cpu"


def trainer(model : nn.Module, Epochs : int, lr : float, 
            trainloader : torch.utils.data.DataLoader, validloader : torch.utils.data.DataLoader, checkpoint_path = "./checkpoint/"):
    
    model = model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=0.0001)
    Train_Losses = []
    Valid_Losses = []

    for epoch in range(Epochs):
        print(f"Epoch {epoch + 1}/{Epochs} ---- ", end='')

        epoch_loss = 0
        valid_loss = 0

        model.train()
        for data in tqdm(trainloader, total=len(trainloader)):
            optimizer.zero_grad()

            xs = data[0].cuda()
            xds = data[1].cuda()
            ys = data[2].cuda()

            output = model(xds, xs)

            loss = criterion(output.squeeze(dim = -1), ys)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for data in tqdm(validloader, total=len(validloader)):
                xs = data[0].cuda()
                xds = data[1].cuda()
                ys = data[2].cuda()

                output = model(xds, xs)

                loss = criterion(output.squeeze(dim = -1), ys)
                valid_loss += loss.item()

        print(f"Train Epoch Loss = {epoch_loss} --------- Validation Epoch Loss = {valid_loss}")
        
        if len(Train_Losses) > 0:
            if valid_loss < min(Valid_Losses):
                torch.save(model.state_dict(), f"{checkpoint_path}best_model.pt")
        elif len(Train_Losses) == 0:
            torch.save(model.state_dict(), f"{checkpoint_path}best_model.pt")

def tester(model : nn.Module, testloader : torch.utils.data.DataLoader, num_companies = 256, num_characteristics = 94):
    
    metric = R2Score()

    model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    outputs = []
    targets = []

    with torch.no_grad():
        test_loss = 0
        for data in tqdm(testloader, total= len(testloader)):
            xs = data[0].cuda()
            xds = data[1].cuda()
            ys = data[2].cuda()

            output = model(xds, xs)

            loss = criterion(output.squeeze(dim = -1), ys)
            metric.update(ys, output.squeeze(dim= -1))
            test_loss += loss.item()

    print(f"Test Loss = {test_loss}")
    print(f"R2 Score = {metric.compute()}")


def predict(model : nn.Module, testloader : torch.utils.data.DataLoader, num_companies : int = 256, num_characteristics : int = 94):
    model.to(device)
    model.eval()

    outputs = []
    targets = []

    with torch.no_grad():
        for data in tqdm(testloader, total= len(testloader)):
            chars = data[1].unsqueeze(dim = 0)
            returns = data[2].unsqueeze(dim = 0).unsqueeze(dim = 2)
            prevReturns = data[3].unsqueeze(dim = 0).unsqueeze(dim = 2)

            if(chars.shape[1] == num_companies):
                output = model(company_characteristics = chars, return_inputs = prevReturns, company_mask = None, return_mask = None)
                
            else:
                chars_new = torch.zeros((1, num_companies - chars.shape[1], num_characteristics), device= device)
                returns_new = torch.zeros((1, num_companies - chars.shape[1], 1), device= device)
                prevReturns_new = torch.zeros((1, num_companies - chars.shape[1],1), device = device)

                chars = torch.cat([chars, chars_new], dim = 1)
                returns = torch.cat([returns, returns_new], dim = 1)
                prevReturns = torch.cat([prevReturns, prevReturns_new], dim = 1)
                
                output = model(company_characteristics = chars, return_inputs = returns, company_mask = None, return_mask = None)

            targets.append(returns)
            outputs.append(output)
    return outputs, targets


