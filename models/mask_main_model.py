import numpy as np
import torch
import torch.nn as nn
from models.mask_diff_models import diff_CSDI, diff_SAITS_new, diff_SAITS_new_2
from datasets.process_data import features

class Mask_base(nn.Module):
    def __init__(self, target_dim, config, device, is_simple=False):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.model_type = config["model"]["type"]
        self.is_simple = is_simple
        self.is_fast = config["diffusion"]['is_fast']
        if config['model']['type'] == 'SAITS':
            ablation_config = config['ablation']

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        if self.model_type == 'SAITS':
            self.is_saits = True
            self.diffmodel = diff_SAITS_new_2(
                diff_steps=config['diffusion']['num_steps'],
                n_layers=config['model']['n_layers'],
                d_time=config['model']['d_time'],
                d_feature=config['model']['n_feature'],
                d_model=config['model']['d_model'],
                d_inner=config['model']['d_inner'],
                n_head=config['model']['n_head'],
                d_k=config['model']['d_k'],
                d_v=config['model']['d_v'],
                dropout=config['model']['dropout'],
                diff_emb_dim=config['diffusion']['diffusion_embedding_dim'],
                diagonal_attention_mask=config['model']['diagonal_attention_mask'],
                is_simple=self.is_simple,
                ablation_config=ablation_config
            )
        else:
            self.is_saits = False
            self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        if self.is_saits:
            self.loss_weight_p = config['model']['loss_weight_p']
            self.loss_weight_f = config['model']['loss_weight_f']

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


    def get_side_info(self, observed_tp, obs_shp):
        B, K, L = obs_shp
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)
        return side_info

    def calc_loss_valid(
        self, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_mask.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_mask)
        noisy_data = (current_alpha ** 0.5) * observed_mask + ((1.0 - current_alpha) ** 0.5) * noise
        total_input = noisy_data.unsqueeze(1)  # (B,1,K,L) 
        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)
        residual = (noise - predicted)
        loss = (residual ** 2).mean()
        return loss

    def impute(self, shape, side_info, n_samples):
        B, K, L = shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            current_sample = torch.randn(shape).to(self.device)
            ti = 0
            for t in range(self.num_steps - 1, -1, -1):
                diff_input = current_sample
                diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)
                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise
                ti += 1
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            _
        ) = self.process_data(batch)
        if self.is_saits:
            side_info = None
        else:
            side_info = self.get_side_info(observed_tp, observed_data.shape)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_mask, side_info, is_train)

    def evaluate(self, n_samples, shape=None, observed_data=None):
        B, K, L = shape if shape is not None else observed_data.shape
        observed_tp = np.expand_dims(np.arange(L), axis=0)
        observed_tp = np.repeat(observed_tp, B, axis=0)
        observed_tp = torch.tensor(observed_tp, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            if self.is_saits:
                side_info = None
            else:
                side_info = self.get_side_info(observed_tp, shape)
            samples = self.impute(shape, side_info, n_samples)
        return samples


class Mask_PM25(Mask_base):
    def __init__(self, config, device, target_dim=36, is_simple=False):
        super(Mask_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            cut_length
        )


class Mask_Physio(Mask_base):
    def __init__(self, config, device, target_dim=35, is_simple=False):
        super(Mask_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        
        return (
            observed_data,
            observed_mask,
            observed_tp,
            cut_length
        )

class Mask_Agaid(Mask_base):
    def __init__(self, config, device, target_dim=len(features), is_simple=False):
        super(Mask_Agaid, self).__init__(target_dim, config, device, is_simple=is_simple)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            cut_length
        )

class Mask_Synth(Mask_base):
    def __init__(self, config, device, target_dim=6, is_simple=False):
        super(Mask_Synth, self).__init__(target_dim, config, device, is_simple=is_simple)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            cut_length
        )

class Mask_AWN(Mask_base):
    def __init__(self, config, device, target_dim=6, is_simple=False):
        super(Mask_AWN, self).__init__(target_dim, config, device, is_simple=is_simple)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch['observed_mask'].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        return (
            observed_data,
            observed_mask,
            observed_tp,
            cut_length
        )
    
class Mask_PM25(Mask_base):
    def __init__(self, config, device, target_dim=6, is_simple=False):
        super(Mask_PM25, self).__init__(target_dim, config, device, is_simple=is_simple)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch['observed_mask'].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        return (
            observed_data,
            observed_mask,
            observed_tp,
            cut_length
        )