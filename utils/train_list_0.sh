CUDA_VISIBLE_DEVICES=0 python -B train.py -verbose -config_file motiongan_v7_action_alldisc_fp_h36
utils/countdown.sh "00:15:00"
CUDA_VISIBLE_DEVICES=0 python -B train.py -verbose -config_file motiongan_v7_action_fp_h36
utils/countdown.sh "00:15:00"
CUDA_VISIBLE_DEVICES=0 python -B train.py -verbose -config_file motiongan_v7_action_coh_fp_h36
utils/countdown.sh "00:15:00"
CUDA_VISIBLE_DEVICES=0 python -B train.py -verbose -config_file motiongan_v7_fp_h36
utils/countdown.sh "00:15:00"
CUDA_VISIBLE_DEVICES=0 python -B train.py -verbose -config_file motiongan_v7_action_nohip_coh_fp_h36em
