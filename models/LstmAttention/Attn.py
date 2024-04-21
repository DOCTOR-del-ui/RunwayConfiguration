import torch
import torch.nn as nn
from math import sqrt

class SingleHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(SingleHeadAttentionLayer, self).__init__()
        self.Q_MATRIX = nn.Linear(d_model, d_k)
        self.K_MATRIX = nn.Linear(d_model, d_k)
        self.V_MATRIX = nn.Linear(d_model, d_v)
        
    @staticmethod
    def scale_attention_score(querys, keys):
        d_k = querys.shape[-1]
        scale_attention_score = (querys @ keys.transpose(1, 2)) / sqrt(d_k)
        return torch.softmax(scale_attention_score, dim=-1)
        
    def forward(self, querys, keys, values):
        B, L_Q, _ = querys.shape
        _, L_KV, _ = keys.shape
        
        querys = self.Q_MATRIX(querys).view(B, L_Q, -1)   # (B, L_Q, d_k)
        keys = self.K_MATRIX(keys).view(B, L_KV, -1)      # (B, L_KV, d_k)
        values = self.V_MATRIX(values).view(B, L_KV, -1)  # (B, H, L_KV, d_v)
        
        attention_score = self.scale_attention_score(querys, keys)
        out = (attention_score @ values)
        return out


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_head = 8, d_model = 512):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.multi_head_attention = nn.ModuleList([
            SingleHeadAttentionLayer(self.d_model, self.d_k, self.d_v)
                for _ in range(self.n_head)
        ])
        self.projection = nn.Linear(self.d_v * self.n_head, d_model)
        
    def forward(self, querys, keys, values):
        heads = []
        for single_head_attention in self.multi_head_attention:
            heads.append(single_head_attention(querys, keys, values))
        out = torch.cat(heads, dim = -1)
        out = self.projection(out)
        return out
    
    
class FeedForward(nn.Module):
    def __init__(self, d_model = 512, hidden = 2048, dropout = 0.3):
        super(FeedForward, self).__init__()
        self.W1 = nn.Linear(d_model, hidden, bias=True)
        self.Relu = nn.ReLU()
        self.W2 = nn.Linear(hidden, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = self.W1(x)
        out = self.Relu(out)
        out = self.W2(out)
        out = self.dropout(out)
        return out
    
    

    

        
        
        
        
        
        
        
        
        
        
        
        
        
        