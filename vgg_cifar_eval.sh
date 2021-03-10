PYTHON="/home/mengjian/anaconda3/bin/python"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
    mkdir ./prob
fi

model=vgg7_quant_eval
dataset=cifar10
batch_size=128
optimizer=SGD
wbit=4
abit=4
mode=dorefa

adc=6
cellbit=2

save_path="./save/${model}/${model}_w${wbit}_a${abit}_mode_${mode}/"
log_file="${model}_w${wbit}_a${abit}_mode${mode}_adc${adc}bit_cell${cellbit}bit.log"
pretrained_model="./save/vgg7_quant/vgg7_quant_w4_a4_mode_dorefa_wd1e-4/model_best.pth.tar"

$PYTHON -W ignore inference.py --dataset ${dataset} \
    --data_path ./dataset/ \
    --model ${model} \
    --save_path ${save_path} \
    --log_file ${log_file} \
    --batch_size ${batch_size} \
    --ngpu 1 \
    --wbit ${wbit} \
    --abit ${abit} \
    --resume ${pretrained_model} \
    --cellBit ${cellbit} \
    --adc_prec ${adc} \
    --fine_tune \
    --evaluate;