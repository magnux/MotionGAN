{
    ## Data Options
    'data_path': './data/',
    # Datasets: MSRC12, NTURGBD
    'data_set': 'NTURGBD',
    'data_set_version': 'v1',
    # Normalize skeletons offline
    'normalize_data': True,
    # Perform per joint normalization
    'normalize_per_joint': False,

    ## Model Options
    # Model type to train
    'model_type': 'motiongan',
    # Model version to train
    'model_version': 'v1',
    # Dropout Rate
    'dropout': 0.0,
    # Lambda for gradient penalty
    'lambda_grads': 10,
    # General loss factor
    'loss_factor': 1.0,
    # Body shape conservation loss
    'shape_loss': False,
    # Body rotation loss
    'rotation_loss': False,
    # Sequence smoothing loss
    'smoothing_loss': False,
    # Disable GAN loss on the generator
    'no_gan_loss': False,
    # Action label conditional model
    'action_cond': False,
    # Latent factor conditional model, size (0 means no condition)
    'latent_cond_dim': 0,
    # Time preserving embedding (NOT COMPATIBLE with latent or FAE)
    'time_pres_emb': False,
    # Use pose FAE
    'use_pose_fae': False,
    # Remove Hip, use hip relative coordinates
    'remove_hip': False,
    # Rescale coords using skeleton average bone len
    'rescale_coords': False,
    # Translate sequence starting point to 0,0,0
    'translate_start': False,
    # Rotate sequence starting point
    'rotate_start': False,
    # Transform euclidean coordinates input to differences
    'use_diff': False,
    # Transform euclidean coordinates input to angles in exponential maps
    'use_angles': False,

    ## DMNN Model Options
    'motiongan_save_path': None,

    ## Training Options
    # It's the batch size
    'batch_size': 256,
    # Final epoch
    'num_epochs': 200,
    # Multiplies length of epoch, useful for tiny datasets
    'epoch_factor': 1,
    # How fast should we learn?
    'learning_rate': 1.0e-3,
    # Train using learning rate decay
    'lr_decay': True,
    # Number of the random picks (0 == deactivated)
    'pick_num': 20,
    # Size of the random crop (0 == deactivated)
    'crop_len': 0,
    # Train on future prediction task only
    'train_fp': False,


    ## Environment Options
    # Random inintializer scale
    'init_scale': 0.1,
    # Load pretrained model
    # 'restore_pretrained': False,
    # Pretrained model path
    # 'pretrained_path': None
}
