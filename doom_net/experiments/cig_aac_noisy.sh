#!/usr/bin/env bash
BASEDIR=$(dirname "$( cd "$( dirname "$0" )" && pwd )")
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [train|resume|test]"
    exit 1
fi

COMMAND="$1"
MODEL=aac_noisy
CHECK_POINT=$BASEDIR/checkpoints/cig_aac_noisy_20_95_skip4_cp.pth
ACTION_SET=$BASEDIR/actions/action_set_speed_shot_backward_right.npy
CONFIG=$BASEDIR/environments/cig.cfg
INSTANCE=cig

if [ $COMMAND == 'train' ]
then
    python $BASEDIR/src/main.py \
    --mode train \
    --episode_size 20 \
    --batch_size 20 \
    --episode_discount 0.95 \
    --model $MODEL \
    --action_set $ACTION_SET \
    --base_model $BASEDIR/trained_models/cig_map01_imitation_model.pth \
    --doom_instance $INSTANCE \
    --vizdoom_config $CONFIG \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 500 \
    --episode_num 150000
elif [ $COMMAND == 'resume' ]
then
    python $BASEDIR/src/main.py \
    --mode train \
    --episode_size 20 \
    --batch_size 20 \
    --episode_discount 0.95 \
    --model $MODEL \
    --action_set $ACTION_SET \
    --load $CHECK_POINT \
    --doom_instance $INSTANCE \
    --vizdoom_config $CONFIG \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 500 \
    --episode_num 150000
elif [ $COMMAND == 'test' ]
then
    python $BASEDIR/src/main.py \
    --mode test \
    --model $MODEL \
    --action_set $ACTION_SET \
    --load $CHECK_POINT \
    --doom_instance $INSTANCE \
    --vizdoom_config $CONFIG \
    --skiprate 1 \
    --frame_num 1
else
    echo "'$COMMAND' is unknown command."
fi