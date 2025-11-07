set -e

source venv/bin/activate

source step0_setup.sh $1

limit=2000
s_limit=200

task_name="gsm8k_adv"

CUDA_VISIBLE_DEVICES=0
thread_idsteer_baseline=0
thread_id=${CUDA_VISIBLE_DEVICES}
task_thread_id=${thread_idsteer_baseline}

model_names=("MyQwen2.5-3B" "gemma-3-4b-it")

layer=-1

# -1: Input embeddings

for model_name in ${model_names[@]}; do
steer_type=steer_baseline
step_source="step2"
step_dest="step3"
input_dir="visualize/${task_name}/${model_name}/${step_source}/${steer_type}"

echo "-------------------------------------"
echo "run baseline"


task_words=($(cat "${input_dir}/word_${layer}.txt"))

    for task_word in ${task_words[@]}; do

      run_sample_gt_fn ${limit} ${task_name} ${task_word} ${task_thread_id} ${model_name} ${step_dest}

    done
done
