import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    

class SoftMapping(pl.LightningModule):
    def __init__(self, input_dim=1024, output_dim=512, lr=1e-4, tau=0.05):
        super().__init__()

        self.attn_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        self.attn_linear = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Dropout(0.5)
        )
        self.lin = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
        self.loss_mse = nn.MSELoss()
        self.lr = lr
        self.tau = tau
        self.log_tau = nn.Parameter(torch.tensor(np.log(tau), dtype=torch.float32))

    def forward(self, x):     # shape: (batch, 200, 1024)
        # attn_weights = torch.softmax(self.attn_linear(x), dim=1)
        attn_weights = torch.sigmoid(self.attn_linear(x))
        attn_out = torch.mean(attn_weights * x, dim=1)
        output = self.lin(attn_out)
        return output, attn_weights

    def cosine_similarity_matrix(self, A, B):
        A_norm = F.normalize(A, dim=1)
        B_norm = F.normalize(B, dim=1)
        return torch.mm(A_norm, B_norm.T)
    
    def ridge_loss(self, y_pred, y_true, alpha):
        mse = F.mse_loss(y_pred, y_true)
        l2_reg = sum(param.pow(2).sum() for param in self.parameters())
        num_params = sum(param.numel() for param in self.parameters())
        l2_reg = l2_reg / num_params
        return mse + alpha * l2_reg

    def contrastive_loss_nt(self, S, tau):
        tau = torch.exp(self.log_tau)  
        S_exp = torch.exp(S / tau)
        loss = -torch.log(torch.diag(S_exp) / S_exp.sum(dim=1))
        return loss.mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        # loss = self.loss_fn(self(x), y)
        output, attn_weights = self(x)
        cos_matrix = self.cosine_similarity_matrix(output, y)
        loss = self.contrastive_loss_nt(cos_matrix, self.tau)
        loss_mse = self.ridge_loss(output, y, 1e-4)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("tau", torch.exp(self.log_tau).item(), prog_bar=True)  
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # loss = self.loss_cl(self(x), y)
        output, attn_weights = self(x)
        cos_matrix = self.cosine_similarity_matrix(output, y)
        loss = self.contrastive_loss_nt(cos_matrix, self.tau)
        loss_mse = self.ridge_loss(output, y, 1e-4)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
 
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)


class TemporalNeuralToFeature(pl.LightningModule):
    def __init__(self, input_dim=1024, hidden_dim=768, output_dim=512, num_layers=1, tau=0.05, lr=1e-4):
        super().__init__()

        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=False)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(output_dim, output_dim)
        )
        self.loss_fn = nn.MSELoss()
        self.tau = tau
        self.lr = lr
        self.log_tau = nn.Parameter(torch.tensor(np.log(self.tau), dtype=torch.float32))

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch, time, hidden)
        final_hidden = rnn_out[:, -1, :]  
        # final_hidden = torch.mean(rnn_out, dim=1)
        return self.mlp(final_hidden)
    
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
        preds = self(x)
        cos_matrix = self.cosine_similarity_matrix(preds, y)
        loss = self.contrastive_loss_nt(cos_matrix, self.tau)
        loss_mse = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("tau", torch.exp(self.log_tau).item(), prog_bar=True)  
        return loss_mse
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        cos_matrix = self.cosine_similarity_matrix(preds, y)
        loss = self.contrastive_loss_nt(cos_matrix, self.tau)
        loss_mse = self.loss_fn(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss_mse

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class SimpleTCN(pl.LightningModule):
    def __init__(self, input_dim=1024, output_dim=512, lr=1e-3, tau=0.05):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.tau = tau
        self.log_tau = nn.Parameter(torch.tensor(np.log(self.tau), dtype=torch.float32))

    def cosine_similarity_matrix(self, A, B):
        A_norm = F.normalize(A, dim=1)
        B_norm = F.normalize(B, dim=1)
        return torch.mm(A_norm, B_norm.T)

    def contrastive_loss_nt(self, S, tau):
        tau = torch.exp(self.log_tau)  
        S_exp = torch.exp(S / tau)
        loss = -torch.log(torch.diag(S_exp) / S_exp.sum(dim=1))
        return loss.mean()

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, 1024, 200)
        x = self.conv(x).squeeze(-1)  # (batch, 128)
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # loss = self.loss_fn(self(x), y)
        cos_matrix = self.cosine_similarity_matrix(self(x), y)
        loss = self.contrastive_loss_nt(cos_matrix, self.tau)
        loss_mse = self.loss_fn(self(x), y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("tau", torch.exp(self.log_tau).item(), prog_bar=True)  
        return loss_mse

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # loss = self.loss_fn(self(x), y)
        cos_matrix = self.cosine_similarity_matrix(self(x), y)
        loss = self.contrastive_loss_nt(cos_matrix, self.tau)
        loss_mse = self.loss_fn(self(x), y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss_mse

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x
    

class TransformerNeuralToFeature(pl.LightningModule):
    def __init__(self, input_dim=1024, d_model=256, nhead=8, num_layers=4, output_dim=512, tau=0.05, lr=1e-3):
        super().__init__()
        
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pooling = nn.AdaptiveAvgPool1d(1)  # mean pooling across time
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.loss_fn = nn.MSELoss()
        self.tau = tau
        self.lr = lr
        self.log_tau = nn.Parameter(torch.tensor(np.log(self.tau), dtype=torch.float32))

    def forward(self, x):  # x: (batch, time, channels)
        x = self.input_proj(x)         # → (batch, time, d_model)
        x = self.pos_encoding(x)       # + positional encodings
        x = self.transformer(x).reshape(x.shape[0], self.d_model, -1)       # → (batch, d_model, time)
        x = self.pooling(x).squeeze(-1)              # mean pooling across time
        return self.mlp(x)             # → (batch, output_dim)
    
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
        preds = self(x)
        loss_mse = self.loss_fn(preds, y)
        cos_matrix = self.cosine_similarity_matrix(preds, y)
        loss = self.contrastive_loss_nt(cos_matrix, self.tau)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("tau", torch.exp(self.log_tau).item(), prog_bar=True) 
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss_mse = self.loss_fn(preds, y)
        cos_matrix = self.cosine_similarity_matrix(preds, y)
        loss = self.contrastive_loss_nt(cos_matrix, self.tau)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    