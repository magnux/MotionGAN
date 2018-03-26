{
    # Normalize skeletons offline
    'normalize_data': False,

    'model_type': 'dmnn',
    # Model version to train
    'model_version': 'v1',

    # It's the batch size
    'batch_size': 32,
    # Final epoch
    'num_epochs': 100,
    # Multiplies length of epoch, useful for tiny datasets
    'epoch_factor': 3,
    # How fast should we learn?
    'learning_rate': 0.5e-4,
    # Dropout Rate
    'dropout': 0.5,

    ## DMNN Model Options
    'motiongan_save_path': 'save/motiongan_v1_tpe',
}