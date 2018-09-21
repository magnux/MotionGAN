CUDA_VISIBLE_DEVICES=1 python -B train.py -verbose -config_file motiongan_v7_action_dmnn_fp_h36
utils/countdown.sh "00:15:00"
CUDA_VISIBLE_DEVICES=1 python -B train.py -verbose -config_file motiongan_v7_action_motion_fp_h36
utils/countdown.sh "00:15:00"
CUDA_VISIBLE_DEVICES=1 python -B train.py -verbose -config_file motiongan_v7_action_nogan_fp_h36
utils/countdown.sh "00:15:00"
CUDA_VISIBLE_DEVICES=1 python -B train.py -verbose -config_file motiongan_v7_h36
utils/countdown.sh "00:15:00"
CUDA_VISIBLE_DEVICES=1 python -B train.py -verbose -config_file motiongan_v7_action_skip_nohip_coh_fp_h36em
utils/countdown.sh "00:15:00"
CUDA_VISIBLE_DEVICES=1 python -B train.py -verbose -config_file motiongan_v7_action_nohip_nogan_fp_h36em