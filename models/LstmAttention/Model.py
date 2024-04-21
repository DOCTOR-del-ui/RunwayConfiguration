import torch
import torch.nn as nn
from models.LstmAttention.Encoder import Encoder
from models.LstmAttention.Decoder import Decoder

class LstmAttention(nn.Module):
    def __init__(self,  n_head = 8, d_model = 512, \
                        ffn_hidden = 2048, max_len = 512, \
                        fea_enc_dim = 9, fea_dec_dim = 15, \
                        num_enc_lstm_layers = 2, num_dec_lstm_layers = 2, \
                        num_enc_layers = 1, num_dec_layers = 1, \
                        out_dim = 27, dropout = 0.3, use_lstm = True, \
                        use_norm1 = True, use_norm2 = True, use_pos = True):
        super(LstmAttention, self).__init__()
        self.encoder = Encoder( n_head, d_model, \
                                ffn_hidden, max_len, \
                                fea_enc_dim, num_enc_lstm_layers, \
                                num_enc_layers, dropout, use_norm1, use_norm2, use_pos)

        self.decoder = Decoder(n_head, d_model, \
                               ffn_hidden, max_len, \
                               fea_dec_dim, num_dec_lstm_layers, \
                               num_dec_layers, dropout, use_norm1, use_norm2, use_pos)
        self.fc1 = nn.Linear(d_model, out_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, enc_input, dec_input):
        enc_output = self.encoder(enc_input)
        dec_output = self.decoder(dec_input, enc_output)
        out = self.fc1(dec_output)
        out = self.softmax(out)
        return out



