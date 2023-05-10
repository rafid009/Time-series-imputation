"""
PyTorch Transformer model for the time-series imputation task.
Some part of the code is from https://github.com/WenjieDu/SAITS.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pypots.data.base import BaseDataset
from pypots.data.dataset_for_mit import DatasetForMIT
from pypots.data.integration import mcar, masked_fill
from pypots.imputation.base import BaseNNImputer
from pypots.utils.metrics import cal_mae


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

def Conv1d_with_init(in_channels, out_channels, kernel_size=1, bias=True, init_zero=False):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
    if init_zero:
        nn.init.zeros_(layer.weight)
    else:
        nn.init.kaiming_normal_(layer.weight)
    return layer

class MultiHeadAttention(nn.Module):
    """original Transformer multi-head attention"""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout, choice='linear', d_channel=-1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.choice = choice
        self.d_model = d_model
        if self.choice == 'fde-conv-single' or self.choice == 'fde-conv-multi':
            self.w_qs = Conv1d_with_init(d_channel, d_channel, kernel_size=1, bias=False)
            self.w_ks = Conv1d_with_init(d_channel, d_channel, kernel_size=1, bias=False)
            self.w_vs = Conv1d_with_init(d_channel, d_channel, kernel_size=1, bias=False)
            if self.choice == 'fde-conv-multi':
                self.w_q_head = Conv1d_with_init(1, self.n_head, kernel_size=1, bias=False)
                self.w_k_head = Conv1d_with_init(1, self.n_head, kernel_size=1, bias=False)
                self.w_v_head = Conv1d_with_init(1, self.n_head, kernel_size=1, bias=False)
        else:
            self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(d_k ** 0.5, attn_dropout)
        

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        if self.choice == 'fde-conv-multi':
            sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
            # q, k, v = q.reshape(-1, len_q, self.d_model), k.reshape(-1, len_k, self.d_model), v.reshape(-1, len_v, self.d_model)
            
            # q = self.w_qs(q).view(sz_b, n_head, len_q, d_k)
            # k = self.w_ks(k).view(sz_b, n_head, len_k, d_k)
            # v = self.w_vs(v).view(sz_b, n_head, len_v, d_v)
            q = self.w_qs(q).view(sz_b * len_q, 1, d_k)
            k = self.w_ks(k).view(sz_b * len_k, 1, d_k)
            v = self.w_vs(v).view(sz_b * len_v, 1, d_v)

            q = self.w_q_head(q).view(sz_b, len_q, n_head, d_k)
            k = self.w_k_head(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_v_head(v).view(sz_b, len_v, n_head, d_v)

            # q = q.permute(0, 2, 1, 3)
            # k = k.permute(0, 2, 1, 3)
            # v = v.permute(0, 2, 1, 3)
        else:
            sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
            q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
            k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, so needs broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)  # For batch and head axis broadcasting.

        v, attn_weights = self.attention(q, k, v, attn_mask)
        # print(f"v after attn: {v.shape}")
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        # print(f"v after attn+fc: {v.shape}")
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, attn_dropout=0.1,
                 diagonal_attention_mask=False, choice='linear', is_ffn=True):
        super().__init__()

        self.diagonal_attention_mask = diagonal_attention_mask
        self.d_time = d_time
        self.d_feature = d_feature

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout, choice=choice, d_channel=d_time)
        self.dropout = nn.Dropout(dropout)
        self.is_ffn = is_ffn
        if self.is_ffn:
            self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)
        else:
            self.pos_ffn = None

    def forward(self, enc_input, mask_time=None):
        if self.diagonal_attention_mask:
            mask_time = torch.eye(self.d_time).to(enc_input.device)
 
        residual = enc_input
        # here we apply LN before attention cal, namely Pre-LN, refer paper https://arxiv.org/abs/2002.04745
        enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=mask_time)
        enc_output = self.dropout(enc_output)
        # print(f"enc_output: {enc_input.shape}")
        enc_output += residual

        if self.is_ffn:
            enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super().__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    @staticmethod
    def _get_sinusoid_encoding_table(n_position, d_hid):
        """ Sinusoid position encoding table """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class _TransformerEncoder(nn.Module):
    def __init__(self, n_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
                 ORT_weight=1, MIT_weight=1):
        super().__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0,
                         False)
            for _ in range(n_layers)
        ])

        self.embedding = nn.Linear(actual_d_feature, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        self.dropout = nn.Dropout(p=dropout)
        self.reduce_dim = nn.Linear(d_model, d_feature)

    def impute(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']
        input_X = torch.cat([X, masks], dim=2)
        input_X = self.embedding(input_X)
        enc_output = self.dropout(self.position_enc(input_X))

        for encoder_layer in self.layer_stack:
            enc_output, _ = encoder_layer(enc_output)

        learned_presentation = self.reduce_dim(enc_output)
        imputed_data = masks * X + (1 - masks) * learned_presentation  # replace non-missing part with original data
        return imputed_data, learned_presentation

    def forward(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']
        imputed_data, learned_presentation = self.impute(inputs)
        reconstruction_loss = cal_mae(learned_presentation, X, masks)

        # have to cal imputation loss in the val stage; no need to cal imputation loss here in the tests stage
        imputation_loss = cal_mae(learned_presentation, inputs['X_intact'], inputs['indicating_mask'])

        loss = self.ORT_weight * reconstruction_loss + self.MIT_weight * imputation_loss

        return {
            'imputed_data': imputed_data,
            'reconstruction_loss': reconstruction_loss, 'imputation_loss': imputation_loss,
            'loss': loss
        }


class Transformer(BaseNNImputer):
    def __init__(self,
                 n_steps,
                 n_features,
                 n_layers,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout,
                 ORT_weight=1,
                 MIT_weight=1,
                 learning_rate=1e-3,
                 epochs=100,
                 patience=10,
                 batch_size=32,
                 weight_decay=1e-5,
                 device=None):
        super().__init__(learning_rate, epochs, patience, batch_size, weight_decay, device)

        self.n_steps = n_steps
        self.n_features = n_features
        # model hype-parameters
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.model = _TransformerEncoder(self.n_layers, self.n_steps, self.n_features, self.d_model, self.d_inner,
                                         self.n_head, self.d_k, self.d_v, self.dropout,
                                         self.ORT_weight, self.MIT_weight)
        self.model = self.model.to(self.device)
        self._print_model_size()

    def fit(self, train_X, val_X=None):
        train_X = self.check_input(self.n_steps, self.n_features, train_X)
        if val_X is not None:
            val_X = self.check_input(self.n_steps, self.n_features, val_X)

        training_set = DatasetForMIT(train_X)
        training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        if val_X is None:
            self._train_model(training_loader)
        else:
            val_X_intact, val_X, val_X_missing_mask, val_X_indicating_mask = mcar(val_X, 0.2)
            val_X = masked_fill(val_X, 1 - val_X_missing_mask, np.nan)
            val_set = DatasetForMIT(val_X)
            val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
            self._train_model(training_loader, val_loader, val_X_intact, val_X_indicating_mask)

        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.
        return self

    def assemble_input_data(self, data):
        """ Assemble the input data into a dictionary.

        Parameters
        ----------
        data : list
            A list containing data fetched from Dataset by Dataload.

        Returns
        -------
        inputs : dict
            A dictionary with data assembled.
        """

        indices, X_intact, X, missing_mask, indicating_mask = data

        inputs = {
            'X': X,
            'X_intact': X_intact,
            'missing_mask': missing_mask,
            'indicating_mask': indicating_mask
        }

        return inputs

    def impute(self, X):
        X = self.check_input(self.n_steps, self.n_features, X)
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = BaseDataset(X)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        imputation_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = {'X': data[1], 'missing_mask': data[2]}
                imputed_data, _ = self.model.impute(inputs)
                imputation_collector.append(imputed_data)

        imputation_collector = torch.cat(imputation_collector)
        return imputation_collector.cpu().detach().numpy()
