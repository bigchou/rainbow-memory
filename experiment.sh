#/bin/bash

# CIL CONFIG
MODE="ewc" # joint, gdumb, icarl, rm, ewc, rwalk, bic
# "default": If you want to use the default memory management method.
MEM_MANAGE="default" # default, random, reservoir, uncertainty, prototype.
RND_SEED=1
DATASET="TinyImagenet" # mnist, cifar10, cifar100, TinyImagenet, imagenet
STREAM="offline" # offline, online
EXP="blurry30" # disjoint, blurry10, blurry30, general10, general30
MEM_SIZE=4000 # cifar10: k={200, 500, 1000}, mnist: k=500, cifar100: k=2,000, imagenet: k=20,000
TRANS="randaug" # multiple choices: cutmix, cutout, randaug, autoaug

N_WORKER=4
JOINT_ACC=0.0 # training all the tasks at once.
# FINISH CIL CONFIG ####################

UNCERT_METRIC="vr_randaug"
PRETRAIN="" INIT_MODEL="" INIT_OPT="--init_opt"

# iCaRL
FEAT_SIZE=128

# BiC
distilling="--distilling" # Normal BiC. If you do not want to use distilling loss, then "".



if [ "$DATASET" == "mnist" ]; then
    TOTAL=50000 N_VAL=250 N_CLASS=10 TOPK=1
    MODEL_NAME="mlp400"
    N_EPOCH=5; BATCHSIZE=16; LR=0.05 OPT_NAME="sgd" SCHED_NAME="cos"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=10 N_CLS_A_TASK=10 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=10 N_CLS_A_TASK=2 N_TASKS=5
    else
        N_INIT_CLS=2 N_CLS_A_TASK=2 N_TASKS=5

    fi
elif [ "$DATASET" == "cifar10" ]; then
    TOTAL=50000 N_VAL=250 N_CLASS=10 TOPK=1
    MODEL_NAME="resnet18"
    N_EPOCH=256; BATCHSIZE=16; LR=0.05 OPT_NAME="sgd" SCHED_NAME="cos"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=10 N_CLS_A_TASK=10 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=10 N_CLS_A_TASK=2 N_TASKS=5
    else
        N_INIT_CLS=2 N_CLS_A_TASK=2 N_TASKS=5

    fi
elif [ "$DATASET" == "cifar100" ]; then
    TOTAL=50000 N_VAL=0 N_CLASS=100 TOPK=1
    MODEL_NAME="resnet18"
    N_EPOCH=256; BATCHSIZE=16; LR=0.03 OPT_NAME="sgd" SCHED_NAME="cos"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=100 N_CLS_A_TASK=100 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=100 N_CLS_A_TASK=20 N_TASKS=5
    else
        N_INIT_CLS=20 N_CLS_A_TASK=20 N_TASKS=5
    fi
elif [ "$DATASET" == "TinyImagenet" ]; then
    TOTAL=100000 N_VAL=0 N_CLASS=200 TOPK=1
    MODEL_NAME="resnet18"
    N_EPOCH=256; BATCHSIZE=64; LR=0.03 OPT_NAME="sgd" SCHED_NAME="cos"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=200 N_CLS_A_TASK=200 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=200 N_CLS_A_TASK=20 N_TASKS=10
    else
        N_INIT_CLS=20 N_CLS_A_TASK=20 N_TASKS=10
    fi 
elif [ "$DATASET" == "imagenet" ]; then
    TOTAL=50000 N_VAL=0 N_CLASS=1000 TOPK=5
    MODEL_NAME="resnet34"
    N_EPOCH=100; BATCHSIZE=256; LR=0.05 OPT_NAME="sgd" SCHED_NAME="multistep"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=1000 N_CLS_A_TASK=100 N_TASKS=10
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=1000 N_CLS_A_TASK=100 N_TASKS=10
    else
        N_INIT_CLS=100 N_CLS_A_TASK=100 N_TASKS=10
    fi
else
    echo "Undefined setting"
    exit 1
fi

python main.py --mode $MODE --mem_manage $MEM_MANAGE --exp_name $EXP \
--dataset $DATASET \
--stream_env $STREAM  $INIT_MODEL $INIT_OPT --topk $TOPK \
--n_tasks $N_TASKS --n_cls_a_task $N_CLS_A_TASK --n_init_cls $N_INIT_CLS \
--rnd_seed $RND_SEED \
--model_name $MODEL_NAME --opt_name $OPT_NAME $PRETRAIN --sched_name $SCHED_NAME \
--lr $LR --batchsize $BATCHSIZE \
--n_worker $N_WORKER --n_epoch $N_EPOCH \
--memory_size $MEM_SIZE --transform $TRANS --uncert_metric $UNCERT_METRIC \
--feature_size $FEAT_SIZE $distilling --joint_acc $JOINT_ACC


#ln -s /home/iis/Desktop/thesis2/dataset/tinyimagenet200 TinyImagenet