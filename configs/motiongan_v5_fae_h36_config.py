{
    # Datasets: MSRC12, NTURGBD
    'data_set': 'Human36',
    'data_set_version': 'v1',
    # Model version to train
    'model_version': 'v5',

    # Use pose FAE
    'use_pose_fae': True,
    # Body shape conservation loss
    'shape_loss': True,
    # Rescale coords using skeleton average bone len
    'rescale_coords': True,
    # Translate sequence starting point to 0,0,0
    'translate_start': True,
    # Rotate sequence starting point
    'rotate_start': True,

    # How fast should we learn?
    'learning_rate': 1.0e-3,
    # It's the batch size
    'batch_size': 256,
    # Multiplies length of epoch, useful for tiny datasets
    'epoch_factor': 200,
    # Number of the random picks (0 == deactivated)
    'pick_num': 20,
    # Size of the random crop (0 == deactivated)
    'crop_len': 100,
}