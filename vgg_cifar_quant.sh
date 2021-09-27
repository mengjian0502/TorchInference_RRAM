PYTHON="/home/mengjian/anaconda3/bin/python"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
    mkdir ./prob
fi

model=vgg7_quant
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
wbit=4
abit=4
mode=sawb

wd=1e-4
lr=0.05

save_path="./save/${model}/${model}_w${wbit}_a${abit}_mode_${mode}_wd${wd}/"
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
    --q_mode ${mode} \
    --a_lambda ${wd} \
    --clp;