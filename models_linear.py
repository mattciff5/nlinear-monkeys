from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class LinearFlatTime(pl.LightningModule):
    def __init__(self, input_dim=1024*200, output_dim=512, lr=1e-3, tau=0.05):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
        self.loss_mse = nn.MSELoss()
        self.lr = lr
        self.tau = tau
        self.log_tau = nn.Parameter(torch.tensor(np.log(tau), dtype=torch.float32))

    def forward(self, x):     # shape: (batch, 200, 1024)
        x = x.reshape(x.shape[0], -1)  
        output = self.linear(x)
        return output

    def cosine_similarity_matrix(self, A, B):
        A_norm = F.normalize(A, dim=1)
        B_norm = F.normalize(B, dim=1)
        return torch.mm(A_norm, B_norm.T)

    def contrastive_loss_nt(self, S, tau):
        tau = torch.exp(self.log_tau)  
        S_exp = torch.exp(S / tau)
        loss = -torch.log(torch.diag(S_exp) / S_exp.sum(dim=1))
        return loss.mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        # loss = self.loss_fn(self(x), y)
        cos_matrix = self.cosine_similarity_matrix(self(x), y)
        loss_cl = self.contrastive_loss_nt(cos_matrix, self.tau)
        mse_loss = self.loss_mse(self(x), y)
        self.log("train_loss", loss_cl, on_epoch=True, prog_bar=True)
        self.log("tau", torch.exp(self.log_tau).item(), prog_bar=True)  
        return mse_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # loss = self.loss_cl(self(x), y)
        cos_matrix = self.cosine_similarity_matrix(self(x), y)
        loss_cl = self.contrastive_loss_nt(cos_matrix, self.tau)
        mse_loss = self.loss_mse(self(x), y)
        self.log("val_loss", loss_cl, on_epoch=True, prog_bar=True)
        return mse_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
    

class MlpAvgTime(pl.LightningModule):
    def __init__(self, input_dim=1024, output_dim=512, lr=1e-4, tau=0.05):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, output_dim)
        )
        self.linear = nn.Sequential(   
            nn.Linear(input_dim, output_dim)
        )
        self.loss_mse = nn.MSELoss()
        self.lr = lr
        self.tau = tau
        self.log_tau = nn.Parameter(torch.tensor(np.log(tau), dtype=torch.float32))

    def forward(self, x):     # shape: (batch, 200, 1024)
        x = x.mean(dim=1)  
        output = self.linear(x)
        return output

    def cosine_similarity_matrix(self, A, B):
        A_norm = F.normalize(A, dim=1)
        B_norm = F.normalize(B, dim=1)
        return torch.mm(A_norm, B_norm.T)

    def contrastive_loss_nt(self, S, tau):
        tau = torch.exp(self.log_tau)  
        S_exp = torch.exp(S / tau)
        loss = -torch.log(torch.diag(S_exp) / S_exp.sum(dim=1))
        return loss.mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        # loss = self.loss_fn(self(x), y)
        cos_matrix = self.cosine_similarity_matrix(self(x), y)
        loss_cl = self.contrastive_loss_nt(cos_matrix, self.tau)
        mse_loss = self.loss_mse(self(x), y)
        self.log("train_loss", loss_cl, on_epoch=True, prog_bar=True)
        self.log("tau", torch.exp(self.log_tau).item(), prog_bar=True)  
        return mse_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # loss = self.loss_cl(self(x), y)
        cos_matrix = self.cosine_similarity_matrix(self(x), y)
        loss_cl = self.contrastive_loss_nt(cos_matrix, self.tau)
        mse_loss = self.loss_mse(self(x), y)
        self.log("val_loss", loss_cl, on_epoch=True, prog_bar=True)
        return mse_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)