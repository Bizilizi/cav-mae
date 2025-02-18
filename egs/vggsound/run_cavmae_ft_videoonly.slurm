#!/bin/bash
#SBATCH --job-name="cav-mae-videoonly-frozen"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --partition=DEADLINE
#SBATCH --comment="CVPR"
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/zverev/logs/cav-mae/slurm-%j.out
#SBATCH --error=/storage/slurm/zverev/logs/cav-mae/slurm-%j.out

# finetune cav-mae pretrained on AS-2M with VGGSound dataset
# you can change pretrain_path to other cav-mae models
source activate cav-mae
set -x

# mount squahshfs
METADATA_DIR=/tmp/$UID/
MOUNTPOINT=$METADATA_DIR/$SLURM_JOBID

trap cleanup EXIT

cleanup () {
fusermount -u $MOUNTPOINT
rmdir $MOUNTPOINT
rmdir /tmp/$UID || true
}


mkdir -p $MOUNTPOINT #create a directory in /tmp
squashfuse /storage/slurm/zverev/datasets/cav-mae.squashfs/cav-mae.squashfs $MOUNTPOINT #mount 

# # comment this line if not running on sls cluster
# . /data/sls/scratch/share-201907/slstoolchainrc
# source /data/sls/scratch/yuangong/avbyol/venv-a5/bin/activate
export TORCH_HOME=../../pretrained_models

model=cav-mae-ft
ftmode=videoonly

# you can replace with any checkpoint you want, but by default, we use cav-mae-scale++
cur_dir=$(pwd)
wget -nc https://www.dropbox.com/s/f4wrbxv2unewss9/vgg_65.5.pth?dl=1 -O cav-mae-scale++.pth
pretrain_path=${cur_dir}/cav-mae-scale++.pth

freeze_base=True
head_lr=10 # newly initialized ft layers uses 10 times larger than the base lr

bal=bal
lr=1e-4
epoch=30
lrscheduler_start=2
lrscheduler_decay=0.5
lrscheduler_step=1
wa=True
wa_start=3
wa_end=10
lr_adapt=False
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
freqm=48
timem=192
mixup=0.5
batch_size=48
label_smooth=0.1

dataset=vggsound

tr_data=/storage/slurm/zverev/datasets/cav-mae/vggsound/metadata/vgg_train_cleaned.json
te_data=/storage/slurm/zverev/datasets/cav-mae/vggsound/metadata/vgg_test_cleaned.json
w_data=/storage/slurm/zverev/datasets/cav-mae/vggsound/metadata/vgg_train_cleaned_weight.csv

# copy train and test files and repalce the path to the mountpoint
cp $tr_data $METADATA_DIR/
cp $te_data $METADATA_DIR/
cp $w_data $METADATA_DIR/

sed -i 's|/storage/slurm/zverev/datasets/cav-mae|'$MOUNTPOINT'|g' $METADATA_DIR/vgg_train_cleaned.json
sed -i 's|/storage/slurm/zverev/datasets/cav-mae|'$MOUNTPOINT'|g' $METADATA_DIR/vgg_test_cleaned.json

tr_data=$METADATA_DIR/vgg_train_cleaned.json
te_data=$METADATA_DIR/vgg_test_cleaned.json

# run the script
exp_dir=./exp/testmae02-${dataset}-${model}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-a5
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_cavmae_ft.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv /storage/slurm/zverev/datasets/cav-mae/vggsound/metadata/class_labels_indices_vgg.csv --n_class 309 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss CE --metrics acc --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--pretrain_path ${pretrain_path} --ftmode ${ftmode} \
--freeze_base ${freeze_base} --head_lr ${head_lr} \
--num-workers 8 \
--n-epochs 10