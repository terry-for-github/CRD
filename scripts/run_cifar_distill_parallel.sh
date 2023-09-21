#!/bin/bash

# sample scripts for running the distillation code
# use resnet32x4 and resnet8x4 as an example
cd /userhome/project/RepDistiller
DATASET=cifar100
TEACHER_ARCHS=('wrn_40_2' 'wrn_40_2' 'resnet56' 'resnet110' 'resnet110' 'resnet32x4' 'vgg13' 'vgg13' 'ResNet50' 'ResNet50' 'resnet32x4' 'resnet32x4' 'wrn_40_2')
STUDENT_ARCHS=('wrn_16_2' 'wrn_40_1' 'resnet20' 'resnet20' 'resnet32' 'resnet8x4' 'vgg8' 'MobileNetV2' 'MobileNetV2' 'vgg8' 'ShuffleV1' 'ShuffleV2' 'ShuffleV1')
# 13
# 0 3 6 9
# 1 4 7 10 12
# 2 5 8 11
KD=(true kd 0.9 0 0.1)
HINT=(true hint 0 100)
AT=(true attention 0 1000)
SP=(true similarity 0 3000)
CC=(true correlation 0 0.02)
VID=(true vid 0 1)
RKD=(true rkd 0 1)
PKT=(true pkt 0 30000)
AB=(true abound 0 0)
FT=(true factor 0 500)
FSP=(false fsp 0 0)
NST=(false nst 0 50)
CRD=(true crd 0 0.8)
CRKD=(true crd 1 0.8)

struct_list=($1)
seed_list=($2)


for i in ${struct_list[@]} ; do
    teacher_arch=${TEACHER_ARCHS[$i]}
    student_arch=${STUDENT_ARCHS[$i]}
    teacher_path=/userhome/download/${teacher_arch}_vanilla/ckpt_epoch_240.pth
    ts_pair=${teacher_arch}-${student_arch}
    log_dir=save/logs/${DATASET}-${ts_pair}

    for t in ${seed_list[@]}; do
        for alg in KD HINT AT SP CC VID RKD PKT AB FT FSP CRD CRKD ; do
            eval "para=(\${$alg[@]})"
            is_enable=${para[0]}
            if ! $is_enable ; then continue ; fi
            name=${para[1]}
            a=${para[2]}
            b=${para[3]}
            if [[ $name = "kd" ]] ; then
                r=${para[4]}
                dir=$log_dir/${name}_a${a}_b${b}_r${r}
                gpu=`python scripts/get_gpu.py -n=${name}_${ts_pair}_$t -t=60`
                # echo "                $gpu $ts_pair $name $a $b $r $t"
                if [ ! -d $dir ] ; then mkdir -p $dir ; fi
                CUDA_VISIBLE_DEVICES=$gpu python train_student.py --path_t $teacher_path --distill $name --model_s $student_arch -a $a -b $b -r $r --trial $t > $dir/$t.log &
                sleep 30
                continue
            fi
            dir=$log_dir/${name}_a${a}_b${b}
            gpu=`python scripts/get_gpu.py -n=${name}_${ts_pair}_$t -t=60`
            # echo "                $gpu $ts_pair $name $a $b $t"
            if [ ! -d $dir ] ; then mkdir -p $dir ; fi
            CUDA_VISIBLE_DEVICES=$gpu python train_student.py --path_t $teacher_path --distill $name --model_s $student_arch -a $a -b $b --trial $t > $dir/$t.log &
            sleep 30
        done
    done
done