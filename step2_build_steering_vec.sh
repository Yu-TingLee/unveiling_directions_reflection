set -e
source venv/bin/activate

source step0_setup.sh $1

s_limit=200
word_limit=5


task_names=("gsm8k_adv" "cruxeval_o_adv")

model_name="MyQwen2.5-3B"

step_source="step0"
step_dest="step2"
wait_token_L2="Wait Alternatively Check"
wait_token_L1="<|endoftext|> % #"
wait_token_L0="Answer Result Output"



for task_name in ${task_names[@]}; do

    input_file="visualize/${task_name}/${step_source}/${task_name}.json"
    output_dir="visualize/${task_name}/${model_name}/${step_dest}/"

    python3 step2_build_steering_vec_batch.py \
        --input_file=${input_file} \
        --model_dir="mymodels/${model_name}" \
        --output_dir="${output_dir}/steer_${s_limit}_20" \
        --wait_token_1 ${wait_token_L2} \
        --wait_token_2 ${wait_token_L0} \
        --limit ${s_limit}

    python3 step2_build_steering_vec_batch.py \
        --input_file=${input_file} \
        --model_dir="mymodels/${model_name}" \
        --output_dir="${output_dir}/steer_${s_limit}_21" \
        --wait_token_1 ${wait_token_L2} \
        --wait_token_2 ${wait_token_L1} \
        --limit ${s_limit}

    python3 step2_build_steering_vec_batch.py \
        --input_file=${input_file} \
        --model_dir="mymodels/${model_name}" \
        --output_dir="${output_dir}/steer_${s_limit}_10" \
        --wait_token_1 ${wait_token_L1} \
        --wait_token_2 ${wait_token_L0} \
        --limit ${s_limit}

    python3 step2_build_steering_vec_batch.py \
        --input_file=${input_file} \
        --is_baseline=1 \
        --model_dir="mymodels/${model_name}" \
        --output_dir="${output_dir}/steer_baseline" \
        --wait_token_1 ${wait_token_L2} \
        --wait_token_2 ""  \
        --limit ${s_limit}

    python3 step2_reselect.py \
        --input_dir="${output_dir}/steer_${s_limit}_21" \
        --word_limit ${word_limit}

    python3 step2_reselect.py \
        --input_dir="${output_dir}/steer_${s_limit}_20" \
        --word_limit ${word_limit}

    python3 step2_reselect.py \
        --input_dir="${output_dir}/steer_${s_limit}_10" \
        --word_limit ${word_limit}

    python3 step2_reselect.py \
        --input_dir="${output_dir}/steer_baseline" \
        --word_limit ${word_limit}

done


model_name="gemma-3-4b-it"

step_source="step0"
step_dest="step2"
wait_token_L2="Wait Alternatively Check"
wait_token_L1="<eos> % #"
wait_token_L0="Answer Result Output"


for task_name in ${task_names[@]}; do

    input_file="visualize/${task_name}/${step_source}/${task_name}.json"
    output_dir="visualize/${task_name}/${model_name}/${step_dest}/"

    python3 step2_build_steering_vec_batch.py \
        --input_file=${input_file} \
        --model_dir="mymodels/${model_name}" \
        --output_dir="${output_dir}/steer_${s_limit}_20" \
        --wait_token_1 ${wait_token_L2} \
        --wait_token_2 ${wait_token_L0} \
        --limit ${s_limit}

    python3 step2_build_steering_vec_batch.py \
        --input_file=${input_file} \
        --model_dir="mymodels/${model_name}" \
        --output_dir="${output_dir}/steer_${s_limit}_21" \
        --wait_token_1 ${wait_token_L2} \
        --wait_token_2 ${wait_token_L1} \
        --limit ${s_limit}

    python3 step2_build_steering_vec.py \
        --input_file=${input_file} \
        --model_dir="mymodels/${model_name}" \
        --output_dir="${output_dir}/steer_${s_limit}_10" \
        --wait_token_1 ${wait_token_L1} \
        --wait_token_2 ${wait_token_L0} \
        --limit ${s_limit}

    python3 step2_build_steering_vec.py \
        --input_file=${input_file} \
        --is_baseline=1 \
        --model_dir="mymodels/${model_name}" \
        --output_dir="${output_dir}/steer_baseline" \
        --wait_token_1 ${wait_token_L2} \
        --wait_token_2 ""  \
        --limit ${s_limit}

    python3 step2_reselect.py \
        --input_dir="${output_dir}/steer_${s_limit}_21" \
        --word_limit ${word_limit}

    python3 step2_reselect.py \
        --input_dir="${output_dir}/steer_${s_limit}_20" \
        --word_limit ${word_limit}

    python3 step2_reselect.py \
        --input_dir="${output_dir}/steer_${s_limit}_10" \
        --word_limit ${word_limit}

    python3 step2_reselect.py \
        --input_dir="${output_dir}/steer_baseline" \
        --word_limit ${word_limit}

done
