#!/bin/bash

# sample scripts for running the distillation code
# use resnet32x4 and resnet8x4 as an example
cd /userhome/project/RepDistiller
TEACHER_PATH=/userhome/download/resnet32x4_vanilla/ckpt_epoch_240.pth
STUDENT_ARCH=resnet8x4
DATASET=cifar100
TS_PAIR=resnet32x4-resnet8x4
LOG_DIR=save/logs/$DATASET-$KD_PAIR

for t in 1 2 3
do

# kd
KD=$LOG_DIR/kd_r0.1_a0.9_b0
CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t $TEACHER_PATH --distill kd --model_s $STUDENT_ARCH -r 0.1 -a 0.9 -b 0 --trial $t > `[[ ! -d "$KD" ]] && mkdir -p $KD;echo $KD/$t.log` &
# FitNet
HINT=$LOG_DIR/hint_a0_b100
CUDA_VISIBLE_DEVICES=2 python train_student.py --path_t $TEACHER_PATH --distill hint --model_s $STUDENT_ARCH -a 0 -b 100 --trial $t > `[[ ! -d "$HINT" ]] && mkdir -p $HINT;echo $HINT/$t.log` &
# AT
AT=$LOG_DIR/attention_a0_b1000
CUDA_VISIBLE_DEVICES=3 python train_student.py --path_t $TEACHER_PATH --distill attention --model_s $STUDENT_ARCH -a 0 -b 1000 --trial $t > `[[ ! -d "$AT" ]] && mkdir -p $AT;echo $AT/$t.log` &
# SP
SP=$LOG_DIR/similarity_a0_b3000
CUDA_VISIBLE_DEVICES=4 python train_student.py --path_t $TEACHER_PATH --distill similarity --model_s $STUDENT_ARCH -a 0 -b 3000 --trial $t > `[[ ! -d "$SP" ]] && mkdir -p $SP;echo $SP/$t.log` &
# CC
CC=$LOG_DIR/correlation_a0_b0.02
CUDA_VISIBLE_DEVICES=5 python train_student.py --path_t $TEACHER_PATH --distill correlation --model_s $STUDENT_ARCH -a 0 -b 0.02 --trial $t  > `[[ ! -d "$CC" ]] && mkdir -p $CC;echo $CC/$t.log` &
# VID
VID=$LOG_DIR/vid_a0_b1
CUDA_VISIBLE_DEVICES=6 python train_student.py --path_t $TEACHER_PATH --distill vid --model_s $STUDENT_ARCH -a 0 -b 1 --trial $t > `[[ ! -d "$VID" ]] && mkdir -p $VID;echo $VID/$t.log` &
# RKD
RKD=$LOG_DIR/rkd_a0_b1
CUDA_VISIBLE_DEVICES=7 python train_student.py --path_t $TEACHER_PATH --distill rkd --model_s $STUDENT_ARCH -a 0 -b 1 --trial $t > `[[ ! -d "$RKD" ]] && mkdir -p $RKD;echo $RKD/$t.log` &

sleep 30
python scripts/get_gpu.py -n='sync' -m=specify -l 1 2 3 4 5 6 7

# PKT
PKT=$LOG_DIR/pkt_a0_b30000
CUDA_VISIBLE_DEVICES=1 python train_student.py --path_t $TEACHER_PATH --distill pkt --model_s $STUDENT_ARCH -a 0 -b 30000 --trial $t > `[[ ! -d "$PKT" ]] && mkdir -p $PKT;echo $PKT/$t.log` &
# AB
AB=$LOG_DIR/abound_a0_b1
CUDA_VISIBLE_DEVICES=2 python train_student.py --path_t $TEACHER_PATH --distill abound --model_s $STUDENT_ARCH -a 0 -b 1 --trial $t > `[[ ! -d "$AB" ]] && mkdir -p $AB;echo $AB/$t.log` &
# FT
FT=$LOG_DIR/factor_a0_b200
CUDA_VISIBLE_DEVICES=3 python train_student.py --path_t $TEACHER_PATH --distill factor --model_s $STUDENT_ARCH -a 0 -b 200 --trial $t > `[[ ! -d "$FT" ]] && mkdir -p $FT;echo $FT/$t.log` &
# FSP
FSP=$LOG_DIR/fsp_a0_b50
CUDA_VISIBLE_DEVICES=4 python train_student.py --path_t $TEACHER_PATH --distill fsp --model_s $STUDENT_ARCH -a 0 -b 50 --trial $t > `[[ ! -d "$FSP" ]] && mkdir -p $FSP;echo $FSP/$t.log` &
# NST
NST=$LOG_DIR/nst_a0_b50
CUDA_VISIBLE_DEVICES=5 python train_student.py --path_t $TEACHER_PATH --distill nst --model_s $STUDENT_ARCH -a 0 -b 50 --trial $t > `[[ ! -d "$NST" ]] && mkdir -p $NST;echo $NST/$t.log` &
# CRD
CRD=$LOG_DIR/crd_a0_b0.8
CUDA_VISIBLE_DEVICES=6 python train_student.py --path_t $TEACHER_PATH --distill crd --model_s $STUDENT_ARCH -a 0 -b 0.8 --trial $t > `[[ ! -d "$CRD" ]] && mkdir -p $CRD;echo $CRD/$t.log` &
# CRD+KD
CRKD=$LOG_DIR/crkd_a1_b0.8
CUDA_VISIBLE_DEVICES=7 python train_student.py --path_t $TEACHER_PATH --distill crd --model_s $STUDENT_ARCH -a 1 -b 0.8 --trial $t > `[[ ! -d "$CRKD" ]] && mkdir -p $CRKD;echo $CRKD/$t.log` &

sleep 30
python scripts/get_gpu.py -n='sync' -m=specify -l 1 2 3 4 5 6 7
done