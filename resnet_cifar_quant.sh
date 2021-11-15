PYTHON="/home/mengjian/anaconda3/bin/python"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
    mkdir ./prob
fi

model=resnet18_quant
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
wbit=4
abit=4
wqmethod=sawb
wqmode=symm

wd=0.0
lambda_swipe=1e-4
lr=0.05
ratio=0.3

# to reduce the weight sensitivity, enable this # to reduce the weight sensitivity, enable --swipe_train

save_path="./save/${model}/${model}_w${wbit}_a${abit}_mode_${wqmethod}_${wqmode}_wd${wd}_lambda_swipe${lambda_swipe}_swipe_train_th/"
log_file="${model}_w${wbit}_a${abit}_mode${mode}_wd${wd}.log"

$PYTHON -W ignore train.py --dataset ${dataset} \
    --data_path ./dataset/ \
    --model ${model} \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --schedule 60 120 \
    --gammas 0.5 0.1 \
    --batch_size ${batch_size} \
    --ngpu 1 \
    --wd ${wd} \
    --wbit ${wbit} \
    --abit ${abit} \
    --q_mode ${wqmethod} \
    --a_lambda ${wd} \
    --wqmode ${wqmode} \
    --lambda_swipe ${lambda_swipe} \
    --clp \
    --reg_ratio ${ratio} \
    --swipe_train; 