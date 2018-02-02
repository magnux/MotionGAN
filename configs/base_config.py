{
    ## Data Options
    'data_path': 'data/',
    # Datasets: HumanEvaI, Human36, NTURGBD
    'data_set': 'NTURGBD',
    'data_set_version': 'v1',
    # Format input as distance matrix
    'format_dm': False,

    ## Model Options
    # Model type to train
    'model_type': 'motiongan',
    # Model version to train
    'model_version': 'v1',
    # Type of joint unfolding
    'unfold': None,
    # Dropout Rate
    'dropout': 0.0,

    ## Training Options
    # It's the batch size
    'batch_size': 256,
    # Final epoch
    'num_epochs': 100,
    # How fast should we learn?
    'learning_rate': 1.0e-3,
    # Train using learning rate decay
    'lr_decay': True,
    # Number of the random picks
    'pick_num': 20,


    ## Environment Options
    # Random inintializer scale
    'init_scale': 0.1,
    # Load pretrained model
    # 'restore_pretrained': False,
    # Pretrained model path
    # 'pretrained_path': None
}
