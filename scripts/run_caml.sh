#!/bin/bash

dt=`date '+%Y%m%d_%H%M%S'`


model='ernie'

bert_lr="5e-4"
other_lr="5e-4"
batch_size=12
epochs=30
patience=5
loss_func="BCE"
gpu=0
data_path="Data/split_data2"
dropout_rate="0.5"
bert_path="../../recommend-huawei/bert_chinese"
ernie_path="../../recommend-huawei/ernie-health-zh"
longformer_path="../../recommend-huawei/Erlangshen-Longformer-110M"


echo "****** hyperparameters *******"
echo "enc_name: CAML"
echo "batch_size: $batch_size"
echo "patience: $patience"
echo "learning_rate: bert_lr $bert_lr other_lr $other_lr"
echo "loss_func: $loss_func"
echo "******************************"

for seed in 0 1 2 3 4; do
     python main.py --model_name CAML --gpu $gpu --bert_path $bert_path \
                    --data_path $data_path --embedding_dim 768 \
                    --dropout_rate $dropout_rate --epochs $epochs --batch_size $batch_size --bert_lr $bert_lr \
                    --other_lr $other_lr --warmup_rate 0.3   --patience $patience --seed $seed \
                    --result_path result --accumulation_steps 1 --loss_func $loss_func --mid_lambda 0 \
                    --label_smooth_lambda 0.0 --hidden_size 768 \
                > result/logs/CAML_seed${seed}_${dt}.log
done
