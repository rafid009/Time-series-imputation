import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.transformer import EncoderLayer, PositionalEncoding
from pypots.imputation import SAITS
import numpy as np
# torch.manual_seed(42)

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv1d_with_init_saits(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    # layer = nn.utils.weight_norm(layer)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv2d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


############################### New Design ################################

# def swish(x):
#     return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out
    

def Conv1d_with_init_saits_new(in_channels, out_channels, kernel_size, init_zero=False, dilation=1):
    padding = dilation * ((kernel_size - 1)//2)
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
    # layer = nn.utils.weight_norm(layer)
    if init_zero:
        nn.init.zeros_(layer.weight)
    else:
        nn.init.kaiming_normal_(layer.weight)
    return layer
    
class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out

def get_output_size(H_in, K, s):
    return int((H_in - K)/s) + 1

def conv_with_init(in_channels, out_channel, kernel_size):
    layer = nn.Conv2d(in_channels, out_channel, kernel_size, stride=2)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class ResidualEncoderLayer(nn.Module):
    def __init__(self, channels, d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
            diffusion_embedding_dim=128, diagonal_attention_mask=True) -> None:
        super().__init__()


        # before combi 2
        # self.enc_layer_1 = EncoderLayer(d_time, actual_d_feature, channels, d_inner, n_head, d_k, d_v, dropout, 0,
        #                  diagonal_attention_mask)
        # combi 2
        self.enc_layer_1 = EncoderLayer(d_time, actual_d_feature, channels, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)
        
        self.enc_layer_2 = EncoderLayer(d_time, actual_d_feature, 2 * channels, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.init_proj = Conv1d_with_init_saits_new(d_model, channels, 1)
        self.conv_layer = Conv1d_with_init_saits_new(channels, 2 * channels, kernel_size=1)

        self.cond_proj = Conv1d_with_init_saits_new(d_model, channels, 1)
        self.conv_cond = Conv1d_with_init_saits_new(channels, 2 * channels, kernel_size=1)


        self.res_proj = Conv1d_with_init_saits_new(channels, d_model, 1)
        self.skip_proj = Conv1d_with_init_saits_new(channels, d_model, 1)

        # self.output_proj = Conv1d_with_init_saits_new(channels, 2 * d_model, 1)
        
        # self.norm = nn.LayerNorm([d_time, d_model])
        # self.post_enc_proj = Conv1d_with_init(channels, 4, 1)



    # new_design
    def forward(self, x, cond, diffusion_emb):
        # x Noise
        # L -> time
        # K -> feature
        B, L, K = x.shape

        x_proj = torch.transpose(x, 1, 2) # (B, K, L)
        x_proj = self.init_proj(x_proj)

        cond = torch.transpose(cond, 1, 2) # (B, K, L)
        cond = self.cond_proj(cond)
        

        diff_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x_proj + diff_proj #+ cond

        # attn1
        y = torch.transpose(y, 1, 2) # (B, L, channels)
        y, attn_weights_1 = self.enc_layer_1(y)
        y = torch.transpose(y, 1, 2)


        y = self.conv_layer(y)
        c_y = self.conv_cond(cond)
        y = y + c_y


        y = torch.transpose(y, 1, 2) # (B, L, 2*channels)
        y, attn_weights_2 = self.enc_layer_2(y)
        y = torch.transpose(y, 1, 2)
 

        y1, y2 = torch.chunk(y, 2, dim=1)
        out = torch.sigmoid(y1) * torch.tanh(y2) # (B, channels, L)

        residual = self.res_proj(out) # (B, K, L)
        residual = torch.transpose(residual, 1, 2) # (B, L, K)

        skip = self.skip_proj(out) # (B, K, L)
        skip = torch.transpose(skip, 1, 2) # (B, L, K)


        attn_weights = (attn_weights_1 + attn_weights_2) / 2 #torch.softmax(attn_weights_1 + attn_weights_2, dim=-1)

        return (x + residual) * math.sqrt(0.5), skip, attn_weights


class diff_SAITS_new(nn.Module):
    def __init__(self, diff_steps, diff_emb_dim, n_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v,
            dropout, diagonal_attention_mask=True, is_simple=False, ablation_config=None):
        super().__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.is_simple = is_simple
        self.d_feature = d_feature
        channels = d_model #int(d_model / 2)
        self.ablation_config = ablation_config
        self.d_time = d_time
        self.n_head = n_head
        
        self.layer_stack_for_first_block = nn.ModuleList([
            ResidualEncoderLayer(channels=channels, d_time=d_time, actual_d_feature=actual_d_feature, 
                        d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                        diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
            for _ in range(n_layers)
        ])
        if self.ablation_config['is_2nd_block']:
            self.layer_stack_for_second_block = nn.ModuleList([
                ResidualEncoderLayer(channels=channels, d_time=d_time, actual_d_feature=actual_d_feature, 
                            d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                            diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
                for _ in range(n_layers)
            ])
            # for operation on measurement dim
            self.embedding_2 = nn.Linear(actual_d_feature, d_model)
            self.reduce_dim_beta = nn.Linear(d_model, d_feature)
            self.reduce_dim_z = nn.Linear(d_model, d_feature)

        self.diffusion_embedding = DiffusionEmbedding(diff_steps, diff_emb_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.position_enc_cond = PositionalEncoding(d_model, n_position=d_time)
        self.position_enc_noise = PositionalEncoding(d_model, n_position=d_time)

        # for operation on time dim
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.embedding_cond = nn.Linear(actual_d_feature, d_model)
        self.reduce_skip_z = nn.Linear(d_model, d_feature)
        
        # self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        # for delta decay factor
        if self.ablation_config['weight_combine']:
            self.weight_combine = nn.Linear(d_feature + d_time, d_feature)
        # combi 2 more layers
        
        
        if self.ablation_config['fde-choice'] == 'fde-conv-single':
            self.mask_conv = Conv1d_with_init_saits_new(2 * self.d_feature, self.d_feature, 1)
            self.layer_stack_for_feature_weights = nn.ModuleList([
                EncoderLayer(d_feature, d_time, d_time, d_inner, 1, d_time, d_time, dropout, 0,
                            self.ablation_config['fde-diagonal'], choice='fde-conv-single')
                for _ in range(self.ablation_config['fde-layers'])
            ])
        elif self.ablation_config['fde-choice'] == 'fde-conv-multi':
            self.mask_conv = Conv1d_with_init_saits_new(2 * self.d_feature, self.d_feature, 1)
            self.layer_stack_for_feature_weights = nn.ModuleList([
                EncoderLayer(d_feature, d_time, d_time, d_inner, n_head, d_time, d_time, dropout, 0,
                            self.ablation_config['fde-diagonal'], choice='fde-conv-multi')
                for _ in range(self.ablation_config['fde-layers'])
            ])
            if self.ablation_config['fde-pos-enc']:
                self.fde_pos_enc = PositionalEncoding(d_time, n_position=d_model)
            # self.expand_head = Conv1d_with_init_saits_new(1, n_head, 1)
        else:
            self.mask_conv = Conv1d_with_init_saits_new(2, 1, 1)
            self.layer_stack_for_feature_weights = nn.ModuleList([
                EncoderLayer(d_feature, d_time, d_time, d_inner, n_head, d_time, d_time, dropout, 0,
                            self.ablation_config['fde-diagonal'])
                for _ in range(self.ablation_config['fde-layers'])
            ])

        

    # ds3
    def forward(self, inputs, diffusion_step):
        # print(f"Entered forward")
        X, masks = inputs['X'], inputs['missing_mask']
        ## making the mask same
        masks[:,1,:,:] = masks[:,0,:,:]
        # B, L, K -> B=batch, L=time, K=feature
        X = torch.transpose(X, 2, 3)
        masks = torch.transpose(masks, 2, 3)
        # Feature Dependency Encoder (FDE): We are trying to get a global feature time-series cross-correlation
        # between features. Each feature's time-series will get global aggregated information from other features'
        # time-series. We also get a feature attention/dependency matrix (feature attention weights) from it.
        if self.ablation_config['is_fde'] and self.ablation_config['is_first']:
            cond_X = X[:,0,:,:] + X[:,1,:,:] # (B, L, K)
            shp = cond_X.shape
            if not self.ablation_config['fde-no-mask']:
                # In one branch, we do not apply the missing mask to the inputs of FDE
                # and in the other we stack the mask with the input time-series for each feature
                # and embed them together to get a masked informed time-series data for each feature.
                cond_X = torch.stack([cond_X, masks[:,1,:,:]], dim=1) # (B, 2, L, K)
                cond_X = cond_X.permute(0, 3, 1, 2) # (B, K, 2, L)
                cond_X = cond_X.reshape(-1, 2 * self.d_feature, self.d_time) # (B, 2*K, L)
                cond_X = self.mask_conv(cond_X) # (B, K, L)
                if self.ablation_config['fde-pos-enc']:
                    cond_X = self.fde_pos_enc(cond_X) # (B, K, L)
            else:
                cond_X = torch.transpose(cond_X, 1, 2) # (B, K, L)

            for feat_enc_layer in self.layer_stack_for_feature_weights:
                cond_X, attn_weights_f = feat_enc_layer(cond_X) # (B, K, L), (B, K, K)

            cond_X = torch.transpose(cond_X, 1, 2)
        else:
            cond_X = X[:,1,:,:]
        # combi 2
        input_X_for_first = torch.cat([cond_X, masks[:,1,:,:]], dim=2)
        input_X_for_first = self.embedding_1(input_X_for_first)

        # cond separate
        noise = input_X_for_first
        cond = torch.cat([X[:,0,:,:], masks[:,0,:,:]], dim=2)
        cond = self.embedding_cond(cond)

        diff_emb = self.diffusion_embedding(diffusion_step)
        pos_cond = self.position_enc_cond(cond)
        
        enc_output = self.dropout(self.position_enc_noise(noise))
        skips_tilde_1 = torch.zeros_like(enc_output)
        for encoder_layer in self.layer_stack_for_first_block:
            # old stable better
            enc_output, skip, _ = encoder_layer(enc_output, pos_cond, diff_emb)
            skips_tilde_1 += skip
        skips_tilde_1 /= math.sqrt(len(self.layer_stack_for_first_block))
        skips_tilde_1 = self.reduce_skip_z(skips_tilde_1)

        if self.ablation_config['is_2nd_block']:
            X_tilde_1 = self.reduce_dim_z(enc_output)

            if self.ablation_config['is_fde_2nd']:
                # Feature attention added
                attn_weights_f = attn_weights_f.squeeze(dim=1)  # namely term A_hat in Eq.
                if len(attn_weights_f.shape) == 4:
                    # if having more than 1 head, then average attention weights from all heads
                    attn_weights_f = torch.transpose(attn_weights_f, 1, 3)
                    attn_weights_f = attn_weights_f.mean(dim=3)
                    attn_weights_f = torch.transpose(attn_weights_f, 1, 2)
                    attn_weights_f = torch.softmax(attn_weights_f, dim=-1)
                # Feature encode for second block
                # cond_X = (cond_X + X[:, 1, :, :])
                X_tilde_1 = (X_tilde_1 + skips_tilde_1) @ attn_weights_f + X[:, 1, :, :] #+ X[:, 0, :, :]#((cond_X + X[:, 1, :, :]) * (1 - masks[:, 1, :, :])) / 2 #cond_X #+ X_tilde_1
            else:
                # Old stable better
                # pre_X_tilde = X_tilde_1
                # if self.ablation_config['is_fde'] and not self.ablation_config['is_first']:
                #     X_tilde_1 = X_tilde_1 + skips_tilde_1
                # else:
                X_tilde_1 = X_tilde_1 + skips_tilde_1 + X[:, 1, :, :] # skips_tilde_1 + X[:, 1, :, :] 
           
            # X_tilde_1 = X_tilde_1 + X[:, 1, :, :]

            # second DMSA block

            if self.ablation_config['is_fde'] and not self.ablation_config['is_first']:
                cond_X = X_tilde_1 #+ X[:,0,:,:] #+ X[:,1,:,:] # (B, L, K)
                if not self.ablation_config['fde-no-mask']:
                    # In one branch, we do not apply the missing mask to the inputs of FDE
                    # and in the other we stack the mask with the input time-series for each feature
                    # and embed them together to get a masked informed time-series data for each feature.
                    cond_X = torch.stack([cond_X, masks[:,1,:,:]], dim=1) # (B, 2, L, K)
                    cond_X = cond_X.permute(0, 3, 1, 2) # (B, K, 2, L)
                    cond_X = cond_X.reshape(-1, 2 * self.d_feature, self.d_time) # (B, 2*K, L)
                    cond_X = self.mask_conv(cond_X) # (B, K, L)
                    if self.ablation_config['fde-pos-enc']:
                        cond_X = self.fde_pos_enc(cond_X) # (B, K, L)
                else:
                    cond_X = torch.transpose(cond_X, 1, 2) # (B, K, L)

                for feat_enc_layer in self.layer_stack_for_feature_weights:
                    cond_X, attn_weights_f = feat_enc_layer(cond_X) # (B, K, L), (B, K, K)

                cond_X = torch.transpose(cond_X, 1, 2)
                cond_X = cond_X + X[:, 1, :, :]
            else:
                cond_X = X_tilde_1 #X[:,1,:,:]

            # before combi 2
            input_X_for_second = torch.cat([cond_X, masks[:,1,:,:]], dim=2)
            input_X_for_second = self.embedding_2(input_X_for_second)
            noise = input_X_for_second

            enc_output = self.position_enc_noise(noise)
            skips_tilde_2 = torch.zeros_like(enc_output)
            for encoder_layer in self.layer_stack_for_second_block:
                enc_output, skip, attn_weights = encoder_layer(enc_output, pos_cond, diff_emb)
                skips_tilde_2 += skip

            # skip_tilde_2
            skips_tilde_2 /= math.sqrt(len(self.layer_stack_for_second_block))
            skips_tilde_2 = self.reduce_dim_beta(skips_tilde_2) #self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(skips_tilde_2)))

            if self.ablation_config['weight_combine']:
                # attention-weighted combine
                attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in Eq.
                if len(attn_weights.shape) == 4:
                    # if having more than 1 head, then average attention weights from all heads
                    attn_weights = torch.transpose(attn_weights, 1, 3)
                    attn_weights = attn_weights.mean(dim=3)
                    attn_weights = torch.transpose(attn_weights, 1, 2)

                combining_weights = torch.sigmoid(
                    self.weight_combine(torch.cat([masks[:, 0, :, :], attn_weights], dim=2))
                )  # namely term eta

                skips_tilde_3 = (1 - combining_weights) * skips_tilde_1 + combining_weights * skips_tilde_2
            else:
                skips_tilde_3 = (skips_tilde_1 + skips_tilde_2) * math.sqrt(0.5)
        if self.ablation_config['is_2nd_block']:
            skips_tilde_1 = torch.transpose(skips_tilde_1, 1, 2)
            skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2)
            skips_tilde_3 = torch.transpose(skips_tilde_3, 1, 2)
        else:
            skips_tilde_3 = torch.transpose(skips_tilde_1, 1, 2)
            skips_tilde_1 = None
            skips_tilde_2 = None

        return skips_tilde_1, skips_tilde_2, skips_tilde_3


###################################### New Incremental One ######################################

class ResidualEncoderLayer_new_2(nn.Module):
    def __init__(self, channels, d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
            diffusion_embedding_dim=128, diagonal_attention_mask=True, ablation_config=None, dilation=1) -> None:
        super().__init__()

        self.time_enc_layer = EncoderLayer(d_time, actual_d_feature, 2 * channels, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)
        self.ablation_config = ablation_config

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.init_proj = Conv1d_with_init_saits_new(d_model, channels, 1)
        self.conv_layer = Conv1d_with_init_saits_new(channels, 2 * channels, kernel_size=1)

        self.cond_proj = Conv1d_with_init_saits_new(d_model, 2 * channels, 1)
        self.conv_cond = Conv1d_with_init_saits_new(2 * channels, 2 * channels, kernel_size=1)


        self.res_proj = Conv1d_with_init_saits_new(channels, d_model, 1)
        self.skip_proj = Conv1d_with_init_saits_new(channels, d_model, 1)

        # self.output_proj = Conv1d_with_init_saits_new(channels, 2 * d_model, 1)

        self.mask_conv = Conv1d_with_init_saits_new(2 * channels, 2 * channels, 1)

    # new_design
    def forward(self, x, cond, diffusion_emb, mask):
        # x Noise
        # L -> time
        # D -> feature
        # channels = D
        B, D, L = x.shape

        x_proj = self.init_proj(x) # (B, D, L)
 
        cond = self.cond_proj(cond) # (B, 2*D, L)
        

        diff_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1) # (B, D, 1)
        y = x_proj + diff_proj # (B, D, L)

        y = self.conv_layer(y) # (B, 2D, L)
        # y = y + cond
        # y, attn_weights_feature = self.feature_encoder(y) # (B, K, L), (B, K, K)

        # y = y + cond
        c_y = self.conv_cond(cond) # (B, 2D, L)
        y = y + c_y


        y = torch.transpose(y, 1, 2) # (B, L, 2*channels)
        y, attn_weights_time = self.time_enc_layer(y) # (B, L, 2*channels), (B, L, L)
        y = torch.transpose(y, 1, 2) # (B, 2*channels, L)


        gate, filter = torch.chunk(y, 2, dim=1)
        out = torch.sigmoid(gate) * torch.tanh(filter) # (B, channels, L)

        residual = self.res_proj(out) # (B, K, L)
        skip = self.skip_proj(out) # (B, K, L)

        # attn_weights = (attn_weights_1 + attn_weights_2) / 2 #torch.softmax(attn_weights_1 + attn_weights_2, dim=-1)

        return (x + residual) * math.sqrt(0.5), skip, attn_weights_time
    

class ResidualEncoderLayer_new_3(nn.Module):
    def __init__(self, channels, d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
            diffusion_embedding_dim=128, diagonal_attention_mask=True, ablation_config=None, dilation=1) -> None:
        super().__init__()

        self.time_enc_layer = EncoderLayer(d_time, actual_d_feature, channels, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)
        self.ablation_config = ablation_config

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        
        self.is_linear = ablation_config['reduce-type'] == 'linear'
        self.is_mask_enc_loop = ablation_config['res-block-mask']
        self.is_fde_loop = ablation_config['is-fde-loop']
        
        if self.is_fde_loop:
            self.feature_enc_layer = EncoderLayer(channels, d_time, d_time, d_inner, n_head, d_time, d_time, dropout, 0,
                                True, choice='fde-conv-multi')

        if self.is_linear:
            if self.is_mask_enc_loop:
                self.init_proj = nn.Linear(d_model + actual_d_feature, channels)
                self.init_cond = nn.Linear(d_model + actual_d_feature, channels)
                self.pos_enc = PositionalEncoding(channels, n_position=d_time)
            else:
                self.init_proj = nn.Linear(d_model, channels)
                self.init_cond = nn.Linear(d_model, channels)
            self.output_proj = nn.Linear(channels, d_model)
        else:
            if self.is_mask_enc_loop:
                self.init_proj = Conv1d_with_init_saits_new(d_model + actual_d_feature, channels, 1)
                self.init_cond = Conv1d_with_init_saits_new(d_model + actual_d_feature, channels, 1)
                self.pos_enc = PositionalEncoding(channels, n_position=d_time)
            else:
                self.init_proj = Conv1d_with_init_saits_new(d_model, channels, 1)
                self.init_cond = Conv1d_with_init_saits_new(d_model, channels, 1)
            self.output_proj = Conv1d_with_init_saits_new(channels, d_model, 1)


    # new_design
    def forward(self, x, cond, diffusion_emb, mask=None):
        # x -> Noise
        # L -> time
        # D -> feature_proj
        # K -> features
        # C -> channels
        # For masking with Conv, D = K and for most cases, we will use D = C, except masking with conv
        B, D, L = x.shape
        if self.is_mask_enc_loop:
            B, K, L = mask.shape

        if self.is_linear:
            x_in = torch.transpose(x, 1, 2) # (B, L, D)
            cond_in = torch.transpose(cond, 1, 2) # (B, L, D)
        else:
            x_in = x # (B, D, L)
            cond_in = cond # (B, D, L)

        if self.is_mask_enc_loop:
            if self.is_linear:
                x_in = torch.cat([x_in, mask], dim=-1) # (B, L, D+K)
                cond_in = torch.cat([cond_in, mask], dim=-1) # (B, L, D+K)
            else:
                # It has to be D = K
                x_in = torch.stack([x_in, mask], dim=1) # (B, 2, D, L)
                x_in = x_in.permute(0, 2, 1, 3) # (B, D, 2, L)
                x_in = x_in.reshape(-1, 2*D, self.d_time) # (B, 2*D, L)

                cond_in = torch.stack([cond_in, mask], dim=1) # (B, 2, D, L)
                cond_in = cond_in.permute(0, 2, 1, 3) # (B, D, 2, L)
                cond_in = cond_in.reshape(-1, 2*D, self.d_time) # (B, 2*D, L)
                
        # C = Channels
        x_proj = self.init_proj(x_in) # (B, C, L) / linear -> (B, L, C)  
        cond = self.init_cond(cond_in) # (B, C, L) / linear -> (B, L, C)

        y = x_proj + cond

        if self.is_linear:
            y = torch.transpose(y, 1, 2) # (B, C, L)

        if self.is_mask_enc_loop:
            y = torch.transpose(y, 1, 2) # (B, L, C)
            y = self.pos_enc(y) # (B, L, C)
            y = torch.transpose(y, 1, 2) # (B, C, L) 

        diff_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1) # (B, C, 1)
        y = y + diff_proj  # (B, C, L)

        if self.is_fde_loop:
            y, _ = self.feature_enc_layer(y) # (B, C, L), (B, C, C)
        
        y = torch.transpose(y, 1, 2) # (B, L, C)
        y, attn_weights_time = self.time_enc_layer(y) # (B, L, C), (B, L, L)

        if not self.is_linear:
            y = torch.transpose(y, 1, 2) # (B, C, L)

        y = self.output_proj(y) # (B, D, L) / linear -> (B, L, D)
        if self.is_linear:
            y = torch.transpose(y, 1, 2) # (B, D, L)
        return y, attn_weights_time


class ResBlockEncDec(nn.Module):
    def __init__(self, cond_emb_channel, in_channels, out_channels, d_time, d_inner, n_head, d_k, d_v, dropout,
            diffusion_embedding_dim=128, diagonal_attention_mask=True, ablation_config=None, dilation=1) -> None:
        super().__init__()

        self.is_fde_loop = ablation_config['is-fde-loop']
        self.ablation_config = ablation_config
        self.time_enc_layer = EncoderLayer(d_time, -1, in_channels, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)
        self.noise_proj_init = Conv1d_with_init_saits_new(in_channels, in_channels, 1)
        self.cond_proj_init = Conv1d_with_init_saits_new(cond_emb_channel, in_channels, 1)
        # self.mid_proj = Conv1d_with_init_saits_new(in_channels, in_channels, 1)
        self.noise_proj_out_1 = Conv1d_with_init_saits_new(in_channels, in_channels, 1)
        self.noise_proj_out_2 = Conv1d_with_init_saits_new(in_channels, out_channels, 1)
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, in_channels)
        if self.is_fde_loop:
            self.feature_enc_layer = EncoderLayer(in_channels, d_time, d_time, d_inner, n_head, d_time, d_time, dropout, 0,
                                True, choice='fde-conv-multi')
        
    def forward(self, x, cond, diffusion_emb):
        B, C, L = x.shape
        y = self.noise_proj_init(x) # (B, C, L)
        cy = self.cond_proj_init(cond) # (B, C, L)
        y = y + cy
        # print(f"y = {y.shape}, cy: {cy.shape}")
        diff_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1) # (B, C, 1)
        # print(f"diff: {diff_emb.shape}")
        y = y + diff_emb # (B, C, L)

        # y = self.mid_proj(y) # (B, C, L)
         # (B, C, L)
        y = torch.transpose(y, 1, 2) # (B, L, C)
        y, attn_time = self.time_enc_layer(y) # (B, L, C), (B, L, L)
        y = torch.transpose(y, 1, 2) # (B, C, L)

        if self.is_fde_loop:
            y, attn_feat = self.feature_enc_layer(y) # (B, C, L), (B, C, C)
        y = self.noise_proj_out_1(y) # (B, C, L)
        y = self.noise_proj_out_2(y) # (B, Cout, L)
        return y


class EncoderDecoderBlock(nn.Module):
    def __init__(self, n_layers, d_model, d_time, d_inner, n_head, d_k, d_v, dropout, 
                 diff_emb_dim, diagonal_attention_mask, ablation_config) -> None:
        super().__init__()
        channels = d_model
        layers = []
        for i in range(n_layers):
            # d_k = channels
            # d_v = channels
            layer = ResBlockEncDec(cond_emb_channel=d_model, in_channels=channels, out_channels=int(channels/2), d_time=d_time, d_inner=d_inner,
                                    n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout, diffusion_embedding_dim=diff_emb_dim, 
                                    diagonal_attention_mask=diagonal_attention_mask, ablation_config=ablation_config)
            layers.append(layer)
            channels //= 2
        self.layer_stack_for_encoder = nn.ModuleList(layers)

        layers = []
        for i in range(n_layers):
            layer = ResBlockEncDec(cond_emb_channel=d_model, in_channels=channels, out_channels=2*channels, d_time=d_time, d_inner=d_inner,
                                    n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout, diffusion_embedding_dim=diff_emb_dim, 
                                    diagonal_attention_mask=diagonal_attention_mask, ablation_config=ablation_config)
            layers.append(layer)
            channels *= 2
        self.layer_stack_for_decoder = nn.ModuleList(layers)

    def forward(self, x, cond, diff_emb):
        skips = []
        y = x
        for i in range(len(self.layer_stack_for_encoder)):
            y = self.layer_stack_for_encoder[i](y, cond, diff_emb)
            if i != len(self.layer_stack_for_encoder) - 1:
                print(f"skips y: {torch.isnan(y).sum()}")
                skips.append(y)
        
        for i in range(len(self.layer_stack_for_decoder)):
            if i != 0:
                print(f"up skips: {torch.isnan(skips[len(self.layer_stack_for_encoder) - i - 1]).sum()}")
                y = (y + skips[len(self.layer_stack_for_encoder) - i - 1])# * math.sqrt(0.5)
                print(f"nan: {torch.isnan(y).sum()}")
            y = self.layer_stack_for_decoder[i](y, cond, diff_emb)
        return y


class diff_SAITS_new_2(nn.Module):
    def __init__(self, diff_steps, diff_emb_dim, n_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v,
            dropout, diagonal_attention_mask=True, is_simple=False, ablation_config=None):
        super().__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.is_simple = is_simple
        self.d_feature = d_feature
        self.ablation_config = ablation_config
        self.d_time = d_time
        self.n_head = n_head

        # if self.ablation_config['embed-type'] != 'linear':
        #     d_model = d_feature
        channels = d_model
        
        self.is_not_residual = self.ablation_config['is-not-residual']

        if self.ablation_config['is_stable'] or self.ablation_config['is_dual']:
            self.layer_stack_for_first_block = nn.ModuleList([
                ResidualEncoderLayer(channels=channels, d_time=d_time, actual_d_feature=actual_d_feature, 
                            d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                            diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
                for _ in range(n_layers)
            ])
            self.layer_stack_for_second_block = nn.ModuleList([
                ResidualEncoderLayer(channels=channels, d_time=d_time, actual_d_feature=actual_d_feature, 
                            d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                            diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
                for _ in range(n_layers)
            ])
        elif self.ablation_config['enc-dec']:
            
            self.enc_dec_block_1 = EncoderDecoderBlock(n_layers=n_layers, d_model=d_model, d_time=d_time, d_inner=d_inner, n_head=n_head,
                                                d_k=d_k, d_v=d_v, dropout=dropout, diff_emb_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask,
                                                ablation_config=ablation_config)
            # self.enc_dec_block_2 = EncoderDecoderBlock(n_layers=n_layers, d_model=d_model, d_time=d_time, d_inner=d_inner, n_head=n_head,
            #                                     d_k=d_k, d_v=d_v, dropout=dropout, diff_emb_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask,
            #                                     ablation_config=ablation_config)
            
        elif self.is_not_residual:
            self.layer_stack_for_first_block = nn.ModuleList([
                ResidualEncoderLayer_new_3(channels=channels, d_time=d_time, actual_d_feature=d_feature, 
                            d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                            diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask, ablation_config=self.ablation_config)
                for i in range(n_layers)
            ])
        else:
            self.layer_stack_for_first_block = nn.ModuleList([
                ResidualEncoderLayer_new_2(channels=channels, d_time=d_time, actual_d_feature=d_feature, 
                            d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                            diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask, ablation_config=self.ablation_config)
                for i in range(n_layers)
            ])

        self.diffusion_embedding = DiffusionEmbedding(diff_steps, diff_emb_dim)
        self.dropout = nn.Dropout(p=dropout)


        if not self.ablation_config['res-block-mask']:
            self.position_enc_cond = PositionalEncoding(d_model, n_position=d_time)
            self.position_enc_noise = PositionalEncoding(d_model, n_position=d_time)
            if self.ablation_config['embed-type'] == 'linear':
                self.embedding_cond = nn.Linear(2 * d_feature, d_model)
                self.embedding_1 = nn.Linear(2 * d_feature, d_model)
            else:
                self.embedding_cond = Conv1d_with_init_saits_new(2 * d_feature, d_model, 1)
                self.embedding_1 = Conv1d_with_init_saits_new(2 * d_feature, d_model, 1)
        else:
            if self.ablation_config['embed-type'] == 'linear':
                self.embedding_cond = nn.Linear(d_feature, d_model)
                self.embedding_1 = nn.Linear(d_feature, d_model)
            else:
                self.embedding_cond = Conv1d_with_init_saits_new(d_feature, d_model, 1)
                self.embedding_1 = Conv1d_with_init_saits_new(d_feature, d_model, 1)

        

        if self.ablation_config['is_2nd_block']:
            if self.ablation_config['embed-type'] == 'linear':
                self.embedding_2 = nn.Linear(2 * d_feature, d_model)
            else:
                self.embedding_2 = Conv1d_with_init_saits_new(d_feature, d_feature, 1)

            if self.ablation_config['reduce-type'] == 'linear':
                if not self.is_not_residual:
                    self.reduce_dim_z = nn.Linear(d_model, d_feature)
                self.reduce_skip_z = nn.Linear(d_model, d_feature)
            else:
                if not self.is_not_residual:
                    self.reduce_dim_z = Conv1d_with_init_saits_new(d_model, d_feature, 1)
                self.reduce_skip_z = Conv1d_with_init_saits_new(d_model, d_feature, 1)

        if self.ablation_config['reduce-type'] == 'linear':
            self.reduce_dim_beta = nn.Linear(d_model, d_feature)
            if self.ablation_config['is_dual']:
                self.reduce_dim_gamma = nn.Linear(d_model, d_feature)
            else:
                self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        else:
            self.reduce_dim_beta = Conv1d_with_init_saits_new(d_model, d_feature, 1)
            self.reduce_dim_gamma = Conv1d_with_init_saits_new(d_feature, d_feature, 1)
        
        # for delta decay factor
        if self.ablation_config['weight_combine']:
            self.weight_combine = nn.Linear(d_feature + d_time, d_feature)
        # combi 2 more layers

        if self.ablation_config['is_fde']:
            if self.ablation_config['fde-choice'] == 'fde-conv-multi':
                self.mask_conv = Conv1d_with_init_saits_new(2 * d_feature, d_feature, 1)
                self.layer_stack_for_feature_weights = nn.ModuleList([
                    EncoderLayer(d_feature, d_time, d_time, d_inner, n_head, d_time, d_time, dropout, 0,
                                True, choice='fde-conv-multi')
                    for _ in range(self.ablation_config['fde-layers'])
                ])
            else:
                self.mask_conv = Conv1d_with_init_saits_new(2 * d_feature, d_feature, 1)
                self.layer_stack_for_feature_weights = nn.ModuleList([
                    EncoderLayer(d_feature, d_time, d_time, d_inner, n_head, d_time, d_time, dropout, 0,
                                True)
                    for _ in range(self.ablation_config['fde-layers'])
                ])
            if self.ablation_config['fde-pos-enc']:
                self.fde_pos_enc = PositionalEncoding(d_time, n_position=d_feature)


    def forward(self, inputs, diffusion_step):
        X, masks = inputs['X'], inputs['missing_mask'] # (B, K, L)
        masks[:,1,:,:] = masks[:,0,:,:] # (B, K, L)
        cond = X[:, 0, :, :] # (B, K, L)
        
        # Feature Dependency Encoder (FDE): We are trying to get a global feature time-series cross-sorrelation
        # between features. Each feature's time-series will get global aggregated information from other features'
        # time-series. We also get a feature attention/dependency matrix (feature attention weights) from it.
        if self.ablation_config['is_fde']:
            noise = X[:,0,:,:] + X[:,1,:,:] # (B, K, L)
            if not self.ablation_config['fde-no-mask']:
                # In one branch, we do not apply the missing mask to the inputs of FDE
                # and in the other we stack the mask with the input time-series for each 
                # feature and embed them together to get a masked informed time-series data 
                # for each feature.
                noise = torch.stack([noise, masks[:,1,:,:]], dim=1) # (B, 2, K, L)
                noise = noise.permute(0, 2, 1, 3) # (B, K, 2, L)
                noise = noise.reshape(-1, 2 * self.d_feature, self.d_time) # (B, 2*K, L)
                noise = self.mask_conv(noise) # (B, K, L)
            if self.ablation_config['fde-pos-enc']:
                noise = self.fde_pos_enc(noise) # (B, K, L)
            for feat_enc_layer in self.layer_stack_for_feature_weights:
                noise, attn_weights_f = feat_enc_layer(noise) # (B, K, L), (B, K, K)
        else:
            noise = X[:, 1, :, :] # (B, K, L)

        if self.ablation_config['enc-dec']:
            if self.ablation_config['embed-type'] == 'linear':
                noise = torch.transpose(noise, 1, 2) # (B, L, K)
                noise = torch.cat([noise, torch.transpose(masks[:, 1, :, :], 1, 2)], dim=-1) # (B, L, 2K)
                noise = self.position_enc_noise(self.embedding_1(noise)) # (B, L, D)
                noise = torch.transpose(noise, 1, 2) # (B, D, L)

                cond = torch.transpose(cond, 1, 2) # (B, L, K)
                cond = torch.cat([cond, torch.transpose(masks[:, 0, :, :], 1, 2)], dim=-1) # (B, L, 2K)
                cond = self.position_enc_cond(self.embedding_cond(cond)) # (B, L, D)
                cond = torch.transpose(cond, 1, 2) # (B, D, L)
            else:
                noise = torch.stack([noise, masks[:, 1, :, :]], dim=1) # (B, 2, K, L)
                noise = noise.permute(0, 2, 1, 3) # (B, K, 2, L)
                noise = noise.reshape(-1, 2 * self.d_feature, self.d_time) # (B, 2*K, L)
                noise = self.embedding_1(noise) # (B,2*K, L)
                noise = torch.transpose(noise, 1, 2)
                noise = self.position_enc_cond(noise) # (B, L, D)
                noise = torch.transpose(noise, 1, 2)

                cond = torch.stack([cond, masks[:, 1, :, :]], dim=1) # (B, 2, K, L)
                cond = cond.permute(0, 2, 1, 3) # (B, K, 2, L)
                cond = cond.reshape(-1, 2 * self.d_feature, self.d_time) # (B, 2*K, L)
                cond = self.embedding_cond(cond) # (B,2*K, L)
                cond = torch.transpose(cond, 1, 2)
                cond = self.position_enc_cond(cond) # (B, L, D)
                cond = torch.transpose(cond, 1, 2)
            diffusion_embed = self.diffusion_embedding(diffusion_step) 
            skips_tilde_2 = self.enc_dec_block_1(noise, cond, diffusion_embed) # (B, D, L)

            # if self.ablation_config['reduce-type'] == 'linear':
            #     skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2)
            # skips_tilde_2 = self.reduce_skip_z(skips_tilde_2)
            # if self.ablation_config['reduce-type'] == 'linear':
            #     skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2)

            # skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2) # (B, L, K)
            # skips_tilde_2 = torch.cat([skips_tilde_2, torch.transpose(masks[:, 1, :, :], 1, 2)], dim=-1) # (B, L, 2K)
            # skips_tilde_2 = self.position_enc_noise(self.embedding_2(skips_tilde_2))
            # skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2) # (B, D, L)

            # skips_tilde_2 = self.enc_dec_block_2(skips_tilde_2, cond, diffusion_embed)
            if self.ablation_config['reduce-type'] == 'linear':
                skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2)

            skips_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(skips_tilde_2)))

            if self.ablation_config['reduce-type'] == 'linear':
                skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2)

            skips_tilde_3 = skips_tilde_2 # (B, K, L)
            skips_tilde_2 = None
            skips_tilde_1 = None
        elif self.ablation_config['is_dual']:
            cond_X = X[:,1,:,:]
            # combi 2
            input_X_for_first = torch.cat([cond_X, masks[:,1,:,:]], dim=2)
            input_X_for_first = self.embedding_1(input_X_for_first)

            # cond separate
            noise = input_X_for_first
            cond = torch.cat([X[:,0,:,:], masks[:,0,:,:]], dim=2)
            cond = self.embedding_cond(cond)

            diff_emb = self.diffusion_embedding(diffusion_step)
            pos_cond = self.position_enc_cond(cond)

            enc_output = self.dropout(self.position_enc_noise(noise))
            skips_tilde_1 = torch.zeros_like(enc_output)
            for encoder_layer in self.layer_stack_for_first_block:
                # old stable better
                enc_output, skip, _ = encoder_layer(enc_output, pos_cond, diff_emb)
                skips_tilde_1 += skip
            skips_tilde_1 /= math.sqrt(len(self.layer_stack_for_first_block))
            skips_tilde_1 = self.reduce_skip_z(skips_tilde_1)
            
            X_tilde_1 = self.reduce_dim_z(enc_output)

            # if self.ablation_config['is_fde_2nd']:
            #     # Feature attention added
            #     attn_weights_f = attn_weights_f.squeeze(dim=1)  # namely term A_hat in Eq.
            #     if len(attn_weights_f.shape) == 4:
            #         # if having more than 1 head, then average attention weights from all heads
            #         attn_weights_f = torch.transpose(attn_weights_f, 1, 3)
            #         attn_weights_f = attn_weights_f.mean(dim=3)
            #         attn_weights_f = torch.transpose(attn_weights_f, 1, 2)
            #         attn_weights_f = torch.softmax(attn_weights_f, dim=-1)
            #     # Feature encode for second block
            #     # cond_X = (cond_X + X[:, 1, :, :])
            #     X_tilde_1 = X_tilde_1 @ attn_weights_f + X[:, 1, :, :] #+ X[:, 0, :, :]#((cond_X + X[:, 1, :, :]) * (1 - masks[:, 1, :, :])) / 2 #cond_X #+ X_tilde_1
            # else:
            # Old stable better
            X_tilde_1 = X_tilde_1 + X[:, 1, :, :] # skips_tilde_1 + X[:, 1, :, :] 

            if self.ablation_config['is_fde']:
                cond_X = X_tilde_1 # X[:,0,:,:] + X[:,1,:,:] # (B, L, K)

                if not self.ablation_config['fde-no-mask']:
                    # In one branch, we do not apply the missing mask to the inputs of FDE
                    # and in the other we stack the mask with the input time-series for each feature
                    # and embed them together to get a masked informed time-series data for each feature.
                    cond_X = torch.stack([cond_X, masks[:,1,:,:]], dim=1) # (B, 2, L, K)
                    cond_X = cond_X.permute(0, 3, 1, 2) # (B, K, 2, L)
                    cond_X = cond_X.reshape(-1, 2 * self.d_feature, self.d_time) # (B, 2*K, L)
                    cond_X = self.mask_conv(cond_X) # (B, K, L)
                else:
                    cond_X = torch.transpose(cond_X, 1, 2) # (B, K, L)

                for feat_enc_layer in self.layer_stack_for_feature_weights:
                    cond_X, attn_weights_f = feat_enc_layer(cond_X) # (B, K, L), (B, K, K)

                cond_X = torch.transpose(cond_X, 1, 2)
            else:
                cond_X = X_tilde_1 #X[:,1,:,:]
            
            # before combi 2
            input_X_for_second = torch.cat([cond_X, masks[:,1,:,:]], dim=2)
            input_X_for_second = self.embedding_2(input_X_for_second)
            noise = input_X_for_second

            enc_output = self.position_enc_noise(noise)
            skips_tilde_2 = torch.zeros_like(enc_output)
            for encoder_layer in self.layer_stack_for_second_block:
                enc_output, skip, attn_weights = encoder_layer(enc_output, pos_cond, diff_emb)
                skips_tilde_2 += skip

            # skip_tilde_2
            skips_tilde_2 /= math.sqrt(len(self.layer_stack_for_second_block))
            skips_tilde_2 =  self.reduce_dim_beta(skips_tilde_2) # self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(skips_tilde_2)))
            enc_output = self.reduce_dim_gamma(enc_output)
            
            if self.ablation_config['weight_combine']:
                # attention-weighted combine
                attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in Eq.
                if len(attn_weights.shape) == 4:
                    # if having more than 1 head, then average attention weights from all heads
                    attn_weights = torch.transpose(attn_weights, 1, 3)
                    attn_weights = attn_weights.mean(dim=3)
                    attn_weights = torch.transpose(attn_weights, 1, 2)

                combining_weights = torch.sigmoid(
                    self.weight_combine(torch.cat([masks[:, 0, :, :], attn_weights], dim=2))
                )  # namely term eta
                skips_tilde_3 = (1 - combining_weights) * skips_tilde_1 + combining_weights * skips_tilde_2
            else:
                skips_tilde_3 = (skips_tilde_1 + skips_tilde_2) * math.sqrt(0.5)

            skips_tilde_1 = torch.transpose(skips_tilde_1, 1, 2)
            skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2)
            skips_tilde_3 = torch.transpose(skips_tilde_3, 1, 2)
            enc_output = torch.transpose(enc_output, 1, 2)

        elif self.ablation_config['is_stable']:
            X = torch.transpose(X, 2, 3)
            masks = torch.transpose(masks, 2, 3)
            # Feature Dependency Encoder (FDE): We are trying to get a global feature time-series cross-correlation
            # between features. Each feature's time-series will get global aggregated information from other features'
            # time-series. We also get a feature attention/dependency matrix (feature attention weights) from it.
            if self.ablation_config['is_fde'] and self.ablation_config['is_first']:
                cond_X = X[:,0,:,:] + X[:,1,:,:] # (B, L, K)
                shp = cond_X.shape
                if not self.ablation_config['fde-no-mask']:
                    # In one branch, we do not apply the missing mask to the inputs of FDE
                    # and in the other we stack the mask with the input time-series for each 
                    # feature and embed them together to get a masked informed time-series data
                    # for each feature.
                    cond_X = torch.stack([cond_X, masks[:,1,:,:]], dim=1) # (B, 2, L, K)
                    cond_X = cond_X.permute(0, 3, 1, 2) # (B, K, 2, L)
                    cond_X = cond_X.reshape(-1, 2 * self.d_feature, self.d_time) # (B, 2*K, L)
                    cond_X = self.mask_conv(cond_X) # (B, K, L)
                else:
                    cond_X = torch.transpose(cond_X, 1, 2) # (B, K, L)

                for feat_enc_layer in self.layer_stack_for_feature_weights:
                    cond_X, attn_weights_f = feat_enc_layer(cond_X) # (B, K, L), (B, K, K)

                cond_X = torch.transpose(cond_X, 1, 2)
            else:
                cond_X = X[:,1,:,:]
            # combi 2
            input_X_for_first = torch.cat([cond_X, masks[:,1,:,:]], dim=2)
            input_X_for_first = self.embedding_1(input_X_for_first)

            # cond separate
            noise = input_X_for_first
            cond = torch.cat([X[:,0,:,:], masks[:,0,:,:]], dim=2)
            cond = self.embedding_cond(cond)

            diff_emb = self.diffusion_embedding(diffusion_step)
            pos_cond = self.position_enc_cond(cond)

            enc_output = self.dropout(self.position_enc_noise(noise))
            skips_tilde_1 = torch.zeros_like(enc_output)
            for encoder_layer in self.layer_stack_for_first_block:
                # old stable better
                enc_output, skip, _ = encoder_layer(enc_output, pos_cond, diff_emb)
                skips_tilde_1 += skip
            skips_tilde_1 /= math.sqrt(len(self.layer_stack_for_first_block))
            skips_tilde_1 = self.reduce_skip_z(skips_tilde_1)
            
            X_tilde_1 = self.reduce_dim_z(enc_output)

            if self.ablation_config['is_fde_2nd']:
                # Feature attention added
                attn_weights_f = attn_weights_f.squeeze(dim=1)  # namely term A_hat in Eq.
                if len(attn_weights_f.shape) == 4:
                    # if having more than 1 head, then average attention weights from all heads
                    attn_weights_f = torch.transpose(attn_weights_f, 1, 3)
                    attn_weights_f = attn_weights_f.mean(dim=3)
                    attn_weights_f = torch.transpose(attn_weights_f, 1, 2)
                    attn_weights_f = torch.softmax(attn_weights_f, dim=-1)
                # Feature encode for second block
                # cond_X = (cond_X + X[:, 1, :, :])
                X_tilde_1 = X_tilde_1 @ attn_weights_f + X[:, 1, :, :] #+ X[:, 0, :, :]#((cond_X + X[:, 1, :, :]) * (1 - masks[:, 1, :, :])) / 2 #cond_X #+ X_tilde_1
            else:
                # Old stable better
                X_tilde_1 = X_tilde_1 + skips_tilde_1 + X[:, 1, :, :] # skips_tilde_1 + X[:, 1, :, :] 
                # X_tilde_1 = X_tilde_1 + X[:, 1, :, :]

            # second DMSA block
            if not self.ablation_config['is_first']:
                if self.ablation_config['is_fde']:
                    cond_X = X_tilde_1 # X[:,0,:,:] + X[:,1,:,:] # (B, L, K)
                    shp = cond_X.shape
                    if not self.ablation_config['fde-no-mask']:
                        # In one branch, we do not apply the missing mask to the inputs of FDE
                        # and in the other we stack the mask with the input time-series for each feature
                        # and embed them together to get a masked informed time-series data for each feature.
                        cond_X = torch.stack([cond_X, masks[:,1,:,:]], dim=1) # (B, 2, L, K)
                        cond_X = cond_X.permute(0, 3, 1, 2) # (B, K, 2, L)
                        cond_X = cond_X.reshape(-1, 2 * self.d_feature, self.d_time) # (B, 2*K, L)
                        cond_X = self.mask_conv(cond_X) # (B, K, L)
                    else:
                        cond_X = torch.transpose(cond_X, 1, 2) # (B, K, L)

                    for feat_enc_layer in self.layer_stack_for_feature_weights:
                        cond_X, attn_weights_f = feat_enc_layer(cond_X) # (B, K, L), (B, K, K)

                    cond_X = torch.transpose(cond_X, 1, 2)
                else:
                    cond_X = X_tilde_1 #X[:,1,:,:]

            # before combi 2
            input_X_for_second = torch.cat([cond_X, masks[:,1,:,:]], dim=2)
            input_X_for_second = self.embedding_2(input_X_for_second)
            noise = input_X_for_second

            enc_output = self.position_enc_noise(noise)
            skips_tilde_2 = torch.zeros_like(enc_output)
            for encoder_layer in self.layer_stack_for_second_block:
                enc_output, skip, attn_weights = encoder_layer(enc_output, pos_cond, diff_emb)
                skips_tilde_2 += skip

            # skip_tilde_2
            skips_tilde_2 /= math.sqrt(len(self.layer_stack_for_second_block))
            skips_tilde_2 =  self.reduce_dim_beta(skips_tilde_2) # self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(skips_tilde_2)))

            if self.ablation_config['weight_combine']:
                # attention-weighted combine
                attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in Eq.
                if len(attn_weights.shape) == 4:
                    # if having more than 1 head, then average attention weights from all heads
                    attn_weights = torch.transpose(attn_weights, 1, 3)
                    attn_weights = attn_weights.mean(dim=3)
                    attn_weights = torch.transpose(attn_weights, 1, 2)

                combining_weights = torch.sigmoid(
                    self.weight_combine(torch.cat([masks[:, 0, :, :], attn_weights], dim=2))
                )  # namely term eta
                skips_tilde_3 = (1 - combining_weights) * skips_tilde_1 + combining_weights * skips_tilde_2
            else:
                skips_tilde_3 = (skips_tilde_1 + skips_tilde_2) * math.sqrt(0.5)

            skips_tilde_1 = torch.transpose(skips_tilde_1, 1, 2)
            skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2)
            skips_tilde_3 = torch.transpose(skips_tilde_3, 1, 2)

        else:
            if not self.ablation_config['res-block-mask']:
                noise = torch.transpose(noise, 1, 2) # (B, L, K)
                noise = torch.cat([noise, torch.transpose(masks[:, 1, :, :], 1, 2)], dim=-1) # (B, L, 2K)
                noise = self.dropout(self.position_enc_noise(self.embedding_1(noise))) # (B, L, D)
                noise = torch.transpose(noise, 1, 2) # (B, D, L)

                if self.ablation_config['embed-type'] == 'linear':
                    cond = torch.transpose(cond, 1, 2) # (B, L, K)
                    cond = torch.cat([cond, torch.transpose(masks[:, 0, :, :], 1, 2)], dim=-1) # (B, L, 2K)
                    cond = self.position_enc_cond(self.embedding_cond(cond)) # (B, L, D)
                    cond = torch.transpose(cond, 1, 2) # (B, D, L)
                else:
                    cond = torch.stack([cond, masks[:, 1, :, :]], dim=1) # (B, 2, K, L)
                    cond = cond.permute(0, 2, 1, 3) # (B, K, 2, L)
                    cond = cond.reshape(-1, 2 * self.d_feature, self.d_time) # (B, 2*K, L)
                    cond = self.embedding_cond(cond) # (B,2*K, L)
                    cond = torch.transpose(cond, 1, 2)
                    cond = self.position_enc_cond(cond) # (B, K, L)
                    cond = torch.transpose(cond, 1, 2)
            else:
                noise = torch.transpose(noise, 1, 2) # (B, L, K)
                noise = F.relu(self.embedding_1(noise)) # (B, L, D)
                noise = torch.transpose(noise, 1, 2) # (B, D, L)

                cond = torch.transpose(cond, 1, 2) # (B, L, K)
                cond = self.embedding_2(cond) # (B, L, D)
                cond = torch.transpose(cond, 1, 2) # (B, D, L)


            diffusion_embed = self.diffusion_embedding(diffusion_step)

            skips_tilde_2 = torch.zeros_like(noise)
            if self.ablation_config['is_2nd_block']:
                skips_tilde_1 = torch.zeros_like(noise)
            enc_output = noise
            i = 0
            layers = len(self.layer_stack_for_first_block)
            if self.is_not_residual:
                # Non residual block layers. Using only self-attention like SAITS
                enc_prev = enc_output
                for encoder in self.layer_stack_for_first_block:
                    i += 1
                    if self.ablation_config['res-block-mask']:
                        enc_output_1, attn_weights = encoder(enc_output, cond, diffusion_embed, masks[:, 1, :, :]) # (B, D, L)
                    else:
                        enc_output_1, attn_weights = encoder(enc_output, cond, diffusion_embed) # (B, D, L)
                    if self.ablation_config['skip-connect-no-res-layer'] and i%2 == 0 and i != layers:
                        enc_output = (enc_prev + enc_output_1) * math.sqrt(0.5)
                        enc_prev = enc_output
                    else:
                        enc_output = enc_output_1

                    if self.ablation_config['is_2nd_block']:
                        if i <= layers/2:
                            skips_tilde_1 += enc_output
                        else:
                            skips_tilde_2 += enc_output
                        if i == layers/2:
                            if self.ablation_config['is_fde'] and self.ablation_config['is_fde_2nd']:
                            # Feature attention added
                                attn_weights_f = attn_weights_f.squeeze(dim=1)  # namely term A_hat in Eq.
                                if len(attn_weights_f.shape) == 4:
                                    # if having more than 1 head, then average attention weights from all heads
                                    attn_weights_f = torch.transpose(attn_weights_f, 1, 3)
                                    attn_weights_f = attn_weights_f.mean(dim=3)
                                    attn_weights_f = torch.transpose(attn_weights_f, 1, 2)
                                    attn_weights_f = torch.softmax(attn_weights_f, dim=-1)
                                if self.ablation_config['reduce-type'] == 'linear':
                                    enc_output = torch.transpose(enc_output, 1, 2)
                                    enc_output = self.reduce_skip_z(enc_output)
                                    enc_output = enc_output @ attn_weights_f + torch.transpose(X[:, 1, :, :], 1, 2)
                                    enc_output = torch.transpose(enc_output, 1, 2)
                                else:
                                    enc_output = self.reduce_skip_z(enc_output)
                                    enc_output = torch.transpose(enc_output, 1, 2)
                                    enc_output = enc_output @ attn_weights_f + torch.transpose(X[:, 1, :, :], 1, 2)
                                    enc_output = torch.transpose(enc_output, 1, 2)
                            else:
                                if self.ablation_config['reduce-type'] == 'linear':
                                    enc_output = torch.transpose(enc_output, 1, 2) # (B, L, D)
                                    enc_output = self.reduce_skip_z(enc_output) + torch.transpose(X[:, 1, :, :], 1, 2) # (B, L, K)
                                    if not self.ablation_config['res-block-mask']:
                                        enc_output = torch.cat([enc_output, torch.transpose(masks[:, 0, :, :], 1, 2)], dim=-1) # (B, L, 2K)
                                    enc_output = self.embedding_2(enc_output) # (B, L, D)
                                    enc_output = torch.transpose(enc_output, 1, 2) # (B, D, L)
                                else:
                                    enc_output = self.reduce_skip_z(enc_output) + X[:, 1, :, :] # (B, K, L)
                                    enc_output = self.embedding_2(enc_output) # (B, D, L)
                            
                            skips_tilde_1 /= math.sqrt(int(len(self.layer_stack_for_first_block)/2))
                            skips_tilde_1 = self.reduce_skip_z(skips_tilde_1) # (B, K, L)
                    else:
                        skips_tilde_2 += enc_output # (B, D, L)

                    if i == layers:
                        # if self.ablation_config['reduce-type'] == 'linear':
                        #     skips_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(skips_tilde_2))) # (B, K, L)
                        # else:
                        if self.ablation_config['is_2nd_block']:
                            skips_tilde_2 /= math.sqrt(int(len(self.layer_stack_for_first_block)/2))
                        else:
                            skips_tilde_2 /= math.sqrt(int(len(self.layer_stack_for_first_block)))
                        if self.ablation_config['reduce-type'] == 'linear':
                            skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2)
                        skips_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(skips_tilde_2)))
                        if self.ablation_config['reduce-type'] == 'linear':
                            skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2)
            else:
                for encoder in self.layer_stack_for_first_block:
                    i += 1
                    if self.ablation_config['res-block-mask']:
                        enc_output, skip, attn_weights = encoder(enc_output, cond, diffusion_embed, masks[:, 1, :, :]) # (B, D, L)
                    else:
                        enc_output, skip, attn_weights = encoder(enc_output, cond, diffusion_embed) # (B, D, L)

                    if self.ablation_config['is_2nd_block']:
                        if i <= layers/2:
                            skips_tilde_1 += skip
                        else:
                            skips_tilde_2 += skip
                        if i == layers/2:
                            if self.ablation_config['is_fde'] and self.ablation_config['is_fde_2nd']:
                            # Feature attention added
                                attn_weights_f = attn_weights_f.squeeze(dim=1)  # namely term A_hat in Eq.
                                if len(attn_weights_f.shape) == 4:
                                    # if having more than 1 head, then average attention weights from all heads
                                    attn_weights_f = torch.transpose(attn_weights_f, 1, 3)
                                    attn_weights_f = attn_weights_f.mean(dim=3)
                                    attn_weights_f = torch.transpose(attn_weights_f, 1, 2)
                                    attn_weights_f = torch.softmax(attn_weights_f, dim=-1)
                                if self.ablation_config['reduce-type'] == 'linear':
                                    enc_output = torch.transpose(enc_output, 1, 2)
                                    enc_output = self.reduce_dim_z(enc_output)
                                    enc_output = enc_output @ attn_weights_f + torch.transpose(X[:, 1, :, :], 1, 2)
                                    enc_output = torch.transpose(enc_output, 1, 2)
                                else:
                                    enc_output = self.reduce_dim_z(enc_output)
                                    enc_output = torch.transpose(enc_output, 1, 2)
                                    enc_output = enc_output @ attn_weights_f + torch.transpose(X[:, 1, :, :], 1, 2)
                                    enc_output = torch.transpose(enc_output, 1, 2)
                            else:
                                if self.ablation_config['reduce-type'] == 'linear':
                                    
                                    enc_output = self.reduce_dim_z(enc_output) + X[:, 1, :, :] # (B, K, L)
                                    enc_output = torch.transpose(enc_output, 1, 2) # (B, L, K)
                                    enc_output = torch.cat([enc_output, torch.transpose(masks[:, 0, :, :], 1, 2)], dim=-1) # (B, L, 2K)
                                    enc_output = self.embedding_2(enc_output) # (B, L, D)
                                    enc_output = torch.transpose(enc_output, 1, 2) # (B, D, L)
                                else:
                                    enc_output = self.reduce_dim_z(enc_output) + X[:, 1, :, :]
                                    enc_output = self.embedding_2(enc_output)
                            
                            # if self.ablation_config['reduce-type'] == 'linear':
                            #     skips_tilde_1 = self.reduce_skip_z(skips_tilde_1) # (B, K, L)
                            # else:
                            skips_tilde_1 /= math.sqrt(int(len(self.layer_stack_for_first_block)/2))
                            skips_tilde_1 = self.reduce_skip_z(skips_tilde_1) # (B, K, L)
                    else:
                        skips_tilde_2 += skip # (B, D, L)

                    if i == layers:
                        # if self.ablation_config['reduce-type'] == 'linear':
                        #     skips_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(skips_tilde_2))) # (B, K, L)
                        # else:
                        if self.ablation_config['is_2nd_block']:
                            skips_tilde_2 /= math.sqrt(int(len(self.layer_stack_for_first_block)/2))
                        else:
                            skips_tilde_2 /= math.sqrt(int(len(self.layer_stack_for_first_block)))
                        skips_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(skips_tilde_2)))

            

            if self.ablation_config['is_2nd_block']:
                if self.ablation_config['weight_combine']:
                    # attention-weighted combine
                    attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in Eq.
                    if len(attn_weights.shape) == 4:
                        # if having more than 1 head, then average attention weights from all heads
                        attn_weights = torch.transpose(attn_weights, 1, 3)
                        attn_weights = attn_weights.mean(dim=3)
                        attn_weights = torch.transpose(attn_weights, 1, 2)

                    combining_weights = torch.sigmoid(
                        self.weight_combine(torch.cat([torch.transpose(masks[:, 0, :, :], 1, 2), attn_weights], dim=2))
                    )  # namely term eta
                    combining_weights = torch.transpose(combining_weights, 1, 2)
                    skips_tilde_3 = (1 - combining_weights) * skips_tilde_1 + combining_weights * skips_tilde_2
                else:
                    # skips_tilde_3 = (skips_tilde_1 + skips_tilde_2) / 2
                    skips_tilde_3 = skips_tilde_2
                    skips_tilde_2 = None
                    skips_tilde_1 = None
            else:
                skips_tilde_3 = skips_tilde_2
                skips_tilde_2 = None
                skips_tilde_1 = None

        return skips_tilde_1, skips_tilde_2, skips_tilde_3