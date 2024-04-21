import torch
import torch.nn as nn

class LstmEmbedding(nn.Module):
    def __init__(self, input_size = 9, d_model = 512, lstm_layers = 2):
        super(LstmEmbedding, self).__init__()
        self.layernorm = nn.LayerNorm(normalized_shape = input_size)
        self.lstm = nn.LSTM(input_size, d_model, lstm_layers, batch_first=True)
        
    def forward(self, x):
        out = self.layernorm(x)
        out, _ = self.lstm(out)
        return out
    
    
class DeNormLstmEmbedding(nn.Module):
    def __init__(self, input_size = 9, d_model = 512, lstm_layers = 2):
        super(DeNormLstmEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_size, d_model, lstm_layers, batch_first=True)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return out
    
    
class PositionEmbedding(nn.Module):
    def __init__(self, d_model = 512, max_len = 512):
        device = torch.device("cuda")
        super(PositionEmbedding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model).to(device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len).to(device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).to(device)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        _, seq_len, _ = x.shape
        return self.encoding[:seq_len, :]
    
    
class Embedding(nn.Module):
    def __init__(self, input_size = 9, d_model = 512, lstm_layers = 2, max_len = 512, dropout = 0.3, use_norm = True, use_pos = True):
        super(Embedding, self).__init__()
        self.use_norm = use_norm
        self.use_pos = use_pos
        self.lstmembedding = LstmEmbedding(input_size, d_model, lstm_layers)
        self.denormlstmembedding = DeNormLstmEmbedding(input_size, d_model, lstm_layers)
        self.positionembedding = PositionEmbedding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if self.use_norm:
            x = self.lstmembedding(x)
        else: 
            x = self.denormlstmembedding(x)
            
        if self.use_pos:
            x = x + self.positionembedding(x)
        return self.dropout(x)