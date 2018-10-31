{
    # Datasets: MSRC12, NTURGBD
    'data_set': 'MSRC12',
    'data_set_version': 'v1',
    # Model version to train
    'model_version': 'v7',

    # Reconstruction loss factor
    'rec_scale': 0.5,
    # Body shape conservation loss
    'shape_loss': True,
    # Shape loss factor
    'shape_scale': 0.05,
    # Augment data on training
    'augment_data': True,
    # Activate dmnn discriminator
    'add_dmnn_disc': True,
    # Activate motion discriminator
    'add_motion_disc': True,

    # How fast should we learn?
    'learning_rate': 1e-3,
    # It's the batch size
    'batch_size': 64,
    # Multiplies length of epoch, useful for tiny datasets
    'epoch_factor': 8,
    # Number of the random picks (0 == deactivated)
    'pick_num': 20,
    # Size of the random crop (0 == deactivated)
    'crop_len': 0,
}