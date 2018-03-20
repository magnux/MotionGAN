{
    # Datasets: MSRC12, NTURGBD
    'data_set': 'MSRC12',
    'data_set_version': 'v1',
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
    'learning_rate': 1.0e-3,
}