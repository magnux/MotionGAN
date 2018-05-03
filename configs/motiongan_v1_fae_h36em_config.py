{
    # Datasets: MSRC12, NTURGBD
    'data_set': 'Human36_expmaps',
    'data_set_version': 'v1',
    # Model version to train
    'model_version': 'v1',

    # Use pose FAE
    'use_pose_fae': True,

    # It's the batch size
    'batch_size': 128,
    # Multiplies length of epoch, useful for tiny datasets
    'epoch_factor': 100,
    # Number of the random picks (0 == deactivated)
    'pick_num': 0,
    # Size of the random crop (0 == deactivated)
    'crop_len': 100,
}