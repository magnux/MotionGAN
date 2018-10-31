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
    # Select GAN type: standard, ralsgan, wgan, no_gan (to disable GAN)
    'gan_type': 'standard',
    # Lambda for gradient penalty
    'lambda_grads': 1.0,
    # GAN loss factor
    'loss_factor': 1.0,
    # Reconstruction loss factor
    'rec_scale': 1.0,
    # Latent conditioning factor size (0 == No latent factor)
    'latent_cond_dim': 0,
    # Latent conditioning loss
    'latent_loss': False,
    # Body shape conservation loss
    'shape_loss': False,
    # Shape loss factor
    'shape_scale': 1.0,
    # Coherence on generated sequences loss
    'coherence_loss': False,
    # Sequence smoothing loss
    'smoothing_loss': False,
    # Action label conditional model
    'action_cond': False,
    # Remove Hip, use hip relative coordinates, incompatible with use_angles
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
    # Augment data on training
    'augment_data': False,
    # Copy last known frame in the input
    'last_known': False,
    # Add skip connection from input to output
    'add_skip': False,
    # Activate dmnn discriminator
    'add_dmnn_disc': False,
    # Activate motion discriminator
    'add_motion_disc': False,

    ## DMNN Model Options
    'motiongan_save_path': None,

    ## Training Options
    # It's the batch size
    'batch_size': 256,
    # Final epoch
    'num_epochs': 128,
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
