{
    # Datasets: MSRC12, NTURGBD
    'data_set': 'MSRC12',
    'data_set_version': 'v1',
    # Model version to train
    'model_version': 'v4',
    # Time preserving embedding (NOT COMPATIBLE with latent factor model)
    'time_pres_emb': True,

    # Final epoch
    'num_epochs': 200,
    # Multiplies length of epoch, useful for tiny datasets
    'epoch_factor': 10,
}