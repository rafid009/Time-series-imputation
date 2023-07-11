import numpy as np
import torch
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

common_config = {
    'n_layers': 4,
    "ablation": {
        "fde-choice": "fde-conv-multi",
        "fde-layers": 4,
        "is_fde": True,
        'weight_combine': True,
        'fde-no-mask': False,
        'fde-diagonal': True,
        'is_fde_2nd': False,
        'fde-pos-enc': True,
        'reduce-type': 'linear',
        'embed-type': 'linear',
        'is_2nd_block': True,
        'is-not-residual': False,
        'res-block-mask': False, 
        'is-fde-loop': False,
        'skip-connect-no-res-layer': False,
        'enc-dec': False,
        'is_stable': True,
        'is_first': True,
        'is_dual': False,
    },
    'name': 'skip_fde_1st_mask_pos_enc_loss_p_bm2'
}

partial_bm_config = {
    'features': 2,
    'length_range': (10, 30),
    'n_chunks': 2
}

def partial_bm(sample, selected_features, length_range, n_chunks):
    length = np.random.randint(length_range[0], length_range[1] + 1)
    k = length
    # mask = ~np.isnan(sample) * 1.0
    length_index = torch.tensor(range(sample.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = np.random.choice(list_of_segments_index, n_chunks, replace=False)
    gt_intact = sample.copy()
    # print(f"feats: {mask[selected_features]}")
    # print(f"snan: {s_nan}")
    # print(f"mask: {mask[selected_features][s_nan[0]:(s_nan[-1] + 1)]}")
    for chunk in range(n_chunks):
        # mask[selected_features][s_nan[chunk][0]:s_nan[chunk][-1] + 1] = 0
        gt_intact[s_nan[chunk][0]:s_nan[chunk][-1] + 1, selected_features] = np.nan
        # print(f"gt: {gt_intact}\ngt_snan: {gt_intact[s_nan[chunk][0]:s_nan[chunk][-1] + 1, selected_features]}")
    obs_data = np.nan_to_num(sample, copy=True)
    mask = ~np.isnan(gt_intact) * 1.0
    # print(f"mask 1: {mask}")
    return obs_data, mask, gt_intact