#!/usr/bin/env bash
set -euo pipefail
source venv/bin/activate

CUDA_VISIBLE_DEVICES="${1:-0}"
export CUDA_VISIBLE_DEVICES NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

MODEL_NAMES=("MyQwen2.5-3B" "MyGemma-3-4B-it")
declare -A HF_ID=(
    [MyQwen2.5-3B]="Qwen/Qwen2.5-3B"
    [MyGemma-3-4B-it]="google/gemma-3-4b-it"
)
declare -A WAIT_TOKEN_L1=(
    [MyQwen2.5-3B]="<|endoftext|> % #"
    [MyGemma-3-4B-it]="<eos> % #"
)
declare -A MODEL_LAYERS=(
    [MyQwen2.5-3B]="$(seq -s ' ' 0 35)"
    [MyGemma-3-4B-it]="$(seq -s ' ' 0 33)"
)

MODELS_DIR="mymodels"
HF_CACHE_DIR=""
VISUALIZE_DIR="visualize"
DATASET_DIR="mydataset"
TASKS_DIR="mytasks"

S_LIMIT=200        # samples for build_vectors.py
WORD_LIMIT=5       # words for reselect_words.py
LIMIT_GSM8K=2000   # samples for lm_eval on gsm8k
LIMIT_CRUXEVAL=500 # samples for lm_eval on cruxeval

WAIT_TOKEN_L2="Wait Alternatively Check"
WAIT_TOKEN_L0="Answer Result Output"

TASK_WORDS_LEVELS=("Wait" "Alternatively" "Check" "<|endoftext|>" "#" "%" "Answer" "Output" "Result")
TASK_NAMES=("gsm8k_adv" "cruxeval_o_adv")

TASK_WORDS_POSITIVE=("<|endoftext|>" "Answer" "#" "%" "Output" "Result")
TASK_WORDS_NEGATIVE=("Wait" "<|endoftext|>" "Alternatively" "Check" "#" "%")
STEER_SUFFIXES_POSITIVE=("20" "21")
STEER_SUFFIXES_NEGATIVE=("20" "10")

TASK_THREAD_ID="${CUDA_VISIBLE_DEVICES}"

CACHE_ARG=()
if [ -n "${HF_CACHE_DIR}" ]; then
    CACHE_ARG=(--cache_dir "${HF_CACHE_DIR}")
fi

# helpers
get_limit() {
    if [ "${1}" = "cruxeval_o_adv" ]; then echo "${LIMIT_CRUXEVAL}"; else echo "${LIMIT_GSM8K}"; fi
}

run_lmeval() {
    local limit=$1 task=$2 out_path=$3 model_name=$4
    local model_args="pretrained=${MODELS_DIR}/${model_name},dtype=float,trust_remote_code=True,local_files_only=True"
    if [ -n "${5:-}" ]; then
        model_args="${model_args},${5}"
    fi
    python3 -m lm_eval --model hf \
        --model_args "${model_args}" \
        --include_path "${TASKS_DIR}" \
        --tasks "${task}" \
        --device cuda:0 \
        --batch_size auto:1 \
        --gen_kwargs max_new_tokens=256 \
        --output_path "${out_path}" \
        --limit "${limit}" \
        --seed 0 \
        --log_samples
}

run_gt() {
    local limit=$1 task_name=$2 task_word=$3 task_thread_id=$4 model_name=$5 step_dest=$6
    local out_path="${VISUALIZE_DIR}/${task_name}/${model_name}/${step_dest}/gt_${limit}_${task_word}"
    echo "${task_word}" > "${TASKS_DIR}/${task_name}/wait_token_${task_thread_id}.txt"
    if [ ! -d "${out_path}" ]; then
        run_lmeval "${limit}" "${task_name}_${task_thread_id}" "${out_path}" "${model_name}"
    fi
}

run_steered() {
    local limit=$1 task_name=$2 task_word=$3 task_thread_id=$4 model_name=$5 step_dest=$6
    local l=$7 c_scale=$8 steer_file=$9 s_suffix=${10}
    local model_args="steering_vec_file=${steer_file},control_type=selected,control_num=${l},control_scale=${c_scale}"
    local out_path="${VISUALIZE_DIR}/${task_name}/${model_name}/${step_dest}/s${c_scale}_${task_word}_${limit}_${s_suffix}_tselected_l${l}"
    echo "${task_word}" > "${TASKS_DIR}/${task_name}/wait_token_${task_thread_id}.txt"
    if [ ! -d "${out_path}" ]; then
        run_lmeval "${limit}" "${task_name}_${task_thread_id}" "${out_path}" "${model_name}" "${model_args}"
    fi
}

build_one_vector() {
    local input_file=$1 hf_id=$2 out_dir=$3; shift 3
    python3 build_vectors.py \
        --input_file="${input_file}" --model_name="${hf_id}" "${CACHE_ARG[@]}" \
        --output_dir="${out_dir}" --limit "${S_LIMIT}" \
        "$@"
}

eval_word_file() {
    local word_file=$1 task_name=$2 model_name=$3 step=$4 max_words=${5:-0}
    [ ! -f "${word_file}" ] && return 0
    local j=0
    while IFS= read -r task_word; do
        [ -z "${task_word}" ] && continue
        if [ "${max_words}" -gt 0 ] && [ "${j}" -ge "${max_words}" ]; then break; fi
        run_gt "${LIMIT_GSM8K}" "${task_name}" "${task_word}" \
            "${TASK_THREAD_ID}" "${model_name}" "${step}"
        j=$((j + 1))
    done < "${word_file}"
}

steer_direction() {
    local model_name=$1 task_name=$2 limit=$3 c_scale=$4 words_var=$5 suffixes_var=$6
    shift 6
    local layers=("$@")
    local words_ref="${words_var}[@]" suffixes_ref="${suffixes_var}[@]"
    for l in "${layers[@]}"; do
        for task_word in "${!words_ref}"; do
            for s_suffix in "${!suffixes_ref}"; do
                local steer_file="${VISUALIZE_DIR}/${task_name}/${model_name}/step2/steer_${S_LIMIT}_${s_suffix}/seed_avg.json"
                run_steered "${limit}" "${task_name}" "${task_word}" "${TASK_THREAD_ID}" \
                    "${model_name}" "step4" "${l}" "${c_scale}" "${steer_file}" "${s_suffix}"
            done
        done
    done
}

# preprocess
for task_name in "${TASK_NAMES[@]}"; do
    python3 preprocess.py \
        --input_file="${DATASET_DIR}/${task_name}/train.json" \
        --json_out_name="${task_name}.json" \
        --visualize_dir="${VISUALIZE_DIR}"
done

# run reflection levels
for model_name in "${MODEL_NAMES[@]}"; do
    for task_word in "${TASK_WORDS_LEVELS[@]}"; do
        for task_name in "${TASK_NAMES[@]}"; do
            run_gt "$(get_limit "${task_name}")" "${task_name}" "${task_word}" \
                "${TASK_THREAD_ID}" "${model_name}" "step1"
        done
    done
done

# build steering vectors
for model_name in "${MODEL_NAMES[@]}"; do
    hf_id="${HF_ID[${model_name}]}"
    wait_token_l1="${WAIT_TOKEN_L1[${model_name}]}"

    for task_name in "${TASK_NAMES[@]}"; do
        input_file="${VISUALIZE_DIR}/${task_name}/step0/${task_name}.json"
        output_dir="${VISUALIZE_DIR}/${task_name}/${model_name}/step2"

        build_one_vector "${input_file}" "${hf_id}" "${output_dir}/steer_${S_LIMIT}_20" \
            --wait_token_1 ${WAIT_TOKEN_L2} --wait_token_2 ${WAIT_TOKEN_L0}
        build_one_vector "${input_file}" "${hf_id}" "${output_dir}/steer_${S_LIMIT}_21" \
            --wait_token_1 ${WAIT_TOKEN_L2} --wait_token_2 ${wait_token_l1}
        build_one_vector "${input_file}" "${hf_id}" "${output_dir}/steer_${S_LIMIT}_10" \
            --wait_token_1 ${wait_token_l1} --wait_token_2 ${WAIT_TOKEN_L0}
        build_one_vector "${input_file}" "${hf_id}" "${output_dir}/steer_baseline" \
            --is_baseline=1 --output_new_vec 0 --wait_token_1 ${WAIT_TOKEN_L2} --wait_token_2 ""

        for steer_type in "steer_${S_LIMIT}_21" "steer_${S_LIMIT}_20" "steer_${S_LIMIT}_10" "steer_baseline"; do
            python3 reselect_words.py \
                --input_dir="${output_dir}/${steer_type}" \
                --word_limit "${WORD_LIMIT}"
        done
    done
done

# instruction selection
TASK_NAME_STEP3="gsm8k_adv"
for model_name in "${MODEL_NAMES[@]}"; do
    read -ra layers <<< "${MODEL_LAYERS[${model_name}]}"
    steer_base="${VISUALIZE_DIR}/${TASK_NAME_STEP3}/${model_name}/step2"

    eval_word_file "${steer_base}/steer_baseline/word_-1.txt" \
        "${TASK_NAME_STEP3}" "${model_name}" "step3"

    for steer_type in "steer_${S_LIMIT}_21" "steer_${S_LIMIT}_20"; do
        for layer in "${layers[@]}"; do
            eval_word_file "${steer_base}/${steer_type}/word_${layer}.txt" \
                "${TASK_NAME_STEP3}" "${model_name}" "step3" 8
        done
    done
done

# activation steering
for model_name in "${MODEL_NAMES[@]}"; do
    read -ra layers <<< "${MODEL_LAYERS[${model_name}]}"

    for task_name in "${TASK_NAMES[@]}"; do
        limit=$(get_limit "${task_name}")

        steer_direction "${model_name}" "${task_name}" "${limit}" "1" \
            TASK_WORDS_POSITIVE STEER_SUFFIXES_POSITIVE "${layers[@]}"
        steer_direction "${model_name}" "${task_name}" "${limit}" "-1" \
            TASK_WORDS_NEGATIVE STEER_SUFFIXES_NEGATIVE "${layers[@]}"
    done
done

# plot results
python3 plot.py \
    --visualize_dir "${VISUALIZE_DIR}" \
    --model_names   "${MODEL_NAMES[@]}" \
    --dataset_names "${TASK_NAMES[@]}" \
    --s_limit       "${S_LIMIT}" \
    --limits        "${LIMIT_GSM8K}" "${LIMIT_CRUXEVAL}"
