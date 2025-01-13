# probabilistic sampling method

import numpy as np
import torch 
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from utility import *

class VariationalLinearModel(nn.Module):
    def __init__(self, n,N):
        super(VariationalLinearModel, self).__init__()
        self.alpha_mu = nn.Parameter(torch.randn(N, 1))  # Mean of alpha
        self.alpha_logvar = nn.Parameter(torch.randn(N, 1))  # Log variance of alpha
        self.b = nn.Parameter(torch.randn(n,1 )) 

    def forward(self, P):
        
        alpha_std = torch.exp(0.5 * self.alpha_logvar)
        alpha = self.alpha_mu + alpha_std * torch.randn_like(alpha_std)
        return torch.matmul(P, alpha) + self.b


def kl_divergence_loss(p, q):
    return kl_divergence(p, q).sum()

def train_model(P, y,  model, learning_rate=0.01, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
    predicted_y_list = []
    loss_list=[]
    
    for epoch in range(epochs):
        model.train()  
        optimizer.zero_grad()  
        
        outputs = model(P)  # Predictions for y (full dataset) of shape (250,1)
        
        outputs_dist = Normal(outputs.mean(), outputs.std())
        y_dist = Normal(y.mean(), y.std())
        kl_loss = kl_divergence_loss(outputs_dist, y_dist)
        loss_list.append(kl_loss.item())
        
        kl_loss.backward()  
        optimizer.step()  
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], KL Loss: {kl_loss.item():.4f}')
        
        predicted_y_list.append(outputs.detach().clone())  

    return  loss_list,predicted_y_list



class Re_VAE_LinearModel(nn.Module):
    def __init__(self, n,N):
        super(Re_VAE_LinearModel, self).__init__()
        self.beta_mu = nn.Parameter(torch.randn(n, 1))  # Mean of beta
        self.beta_logvar = nn.Parameter(torch.randn(n, 1))  # Log variance of beta
        self.b = nn.Parameter(torch.randn(N,1 )) 

    def forward(self, P):
        
        beta_std = torch.exp(0.5 * self.beta_logvar)
        beta = self.beta_mu + beta_std * torch.randn_like(beta_std)
        return torch.matmul(P.T, beta) - self.b





    






    






