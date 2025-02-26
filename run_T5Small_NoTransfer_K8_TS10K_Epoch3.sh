source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"

# Base paths and settings
initial_model="google-t5/t5-small"
set -e

# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local pretrained_model_path=$2
    local json_input=$3
    local json_output=$4
    local reward_label_output=$5
    local output_model_path=$6
    local win_rate_output_dir=$7
    local prev_win_rate_output_dir=$8
    local seed=$9

    conda activate rlhflow
    if [ ! -f $json_output ]; then
        echo 'start to generate responses'
        CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 25500 generation/gen_hf_local.py --eos_ids 128009 --tokenizer $pretrained_model_path --pretrained_model $pretrained_model_path --dataset_name_or_path $json_input --dataset_key document --output_dir $json_output --K 8 --N 0 --temperature 1.0 --batch_size 8 --train_data_size 10000 --dtype bfloat16 --set_eval True --seed $seed
    else
        echo 'detect generated responses, will skip this step'
    fi
    
    if [ ! -f $reward_label_output ]; then
        conda activate vllm
        echo 'start to generate reward labels'
        CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 25500 annotate_data/Annotation.py --dataset_name_or_path $json_output --output_dir $reward_label_output --prev_win_rate_output_dir $prev_win_rate_output_dir --win_rate_output_dir $win_rate_output_dir --K 8 --Source_RMs_Names null --Source_LLMs_Names null --judge_model_tokenizer meta-llama/Meta-Llama-3-8B-Instruct --judge_model_path meta-llama/Meta-Llama-3-8B-Instruct --set_eval True --seed $[$seed + 123]
    else
        echo 'detect generated reward labels, will skip this step'
    fi

    conda activate rlhflow
    if [ ! -d $output_model_path ]; then
        echo 'start to train'
        
        CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 25500 --config_file ./configs/zero2_4GPU.yaml dpo_iteration/run_dpo.py --run_name $output_model_path --output_dir $output_model_path --tokenizer $pretrained_model_path --model_name_or_path $pretrained_model_path --ref_model $pretrained_model_path --learning_rate 5e-5 --max_steps 1200 --choose_type max_min --train_dir $reward_label_output --eval_dir $reward_label_output --loss_type sigmoid --lr_scheduler_type cosine --use_deepspeed False --is_encoder_decoder True --algorithm DPO --total_train_batch_size 64 --train_data_size 10000 --dtype bfloat16  --seed $[$seed + 123456] --num_train_epochs 3
    else
        echo 'detech trained model directory, will skip this step'
    fi

}

for seed in 100 200 300
do
    # Base paths and settings
    parent_path="YOUR_PARENT_PATH/K8_Seed${seed}"
    base_path="${parent_path}/DPO_NoTransfer_K8_TS10K_Ep3"
    mkdir -p $base_path

    # Main loop for iterations
    for i in {1..3}
    do
        iteration_name="Iter${i}"
        json_input="EdinburghNLP/xsum"
        json_output="${base_path}/${iteration_name}.json"
        reward_label_output="${base_path}/${iteration_name}_reward.json"
        output_model_path="${base_path}/${iteration_name}_model_DPO"
        
        # Determine the model path: first iteration uses the initial model, subsequent iterations use the previous iteration's model
        if [ $i -eq 1 ]; then
            pretrained_model_path=$initial_model
            win_rate_output_dir="${base_path}/${iteration_name}_win_rate.json"
            prev_win_rate_output_dir="None"
        else
            var=`expr $i - 1`
            previous_iteration="Iter${var}"
            pretrained_model_path="${base_path}/${previous_iteration}_model_DPO"
            
            prev_win_rate_output_dir="${base_path}/${previous_iteration}_win_rate.json"
            win_rate_output_dir="${base_path}/${iteration_name}_win_rate.json"
        fi
        
        # XXX: make sure the seed is different for each iteration,
        # otherwise, the training batch samples may be the same.
        run_iteration $i $pretrained_model_path $json_input $json_output $reward_label_output $output_model_path $win_rate_output_dir $prev_win_rate_output_dir $[$seed+$i]

    done
done
