common_config = {
    'n_layers': 4,
    "ablation": {
        "fde-choice": "fde-conv-multi",
        "fde-layers": 4,
        "is_fde": True,
        'weight_combine': True,
        'fde-no-mask': True,
        'fde-diagonal': False,
        'is_fde_2nd': True,
        'fde-pos-enc': False,
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
        'is_dual': False
    } 
}