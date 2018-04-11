{
    # Datasets: MSRC12, NTURGBD
    'data_set': 'Human36',
    'data_set_version': 'v1',
    # Model version to train
    'model_version': 'v1',

    # Time preserving embedding (NOT COMPATIBLE with latent factor model)
    'time_pres_emb': True,

    # It's the batch size
    'batch_size': 10,
    # Final epoch
    'num_epochs': 200,
    # Multiplies length of epoch, useful for tiny datasets
    'epoch_factor': 100,
    # How fast should we learn?
    'learning_rate': 1.0e-3,
    # Train using learning rate decay
    'lr_decay': True,
    # Number of the random picks (0 == deactivated)
    'pick_num': 0,
    # Size of the random crop (0 == deactivated)
    'crop_len': 40,
}