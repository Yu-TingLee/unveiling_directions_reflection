#!/usr/bin/env bash
set -euo pipefail
source venv/bin/activate

CUDA_VISIBLE_DEVICES="${1:-0}"
export CUDA_VISIBLE_DEVICES NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

QWEN_HF_ID="Qwen/Qwen2.5-3B"
GEMMA_HF_ID="google/gemma-3-4b-it"
QWEN_MODEL_NAME="MyQwen2.5-3B"
GEMMA_MODEL_NAME="MyGemma-3-4B-it"
MODEL_NAMES=("${QWEN_MODEL_NAME}" "${GEMMA_MODEL_NAME}")
HF_IDS=("${QWEN_HF_ID}"          "${GEMMA_HF_ID}")

MODELS_DIR="mymodels"
HF_CACHE_DIR=""
VISUALIZE_DIR="visualize"
DATASET_DIR="mydataset"
TASKS_DIR="mytasks"

S_LIMIT=200
WORD_LIMIT=5
LIMIT_GSM8K=2000
LIMIT_CRUXEVAL=500

WAIT_TOKEN_L2="Wait Alternatively Check"
WAIT_TOKEN_L1_QWEN="<|endoftext|> % #"
WAIT_TOKEN_L1_GEMMA="<eos> % #"
WAIT_TOKEN_L0="Answer Result Output"

TASK_WORDS_LEVELS=("Wait" "Alternatively" "Check" "<|endoftext|>" "#" "%" "Answer" "Output" "Result")
TASK_NAMES=("gsm8k_adv" "cruxeval_o_adv")

LAYERS_QWEN=(0 1 2 4 6 8 10 12 14 16 18 20 22 25 28 31 33 35)
LAYERS_GEMMA=(0 1 2 4 6 8 10 12 14 16 18 20 22 25 28 31 33)

TASK_WORDS_POSITIVE=("<|endoftext|>" "Answer" "#" "%" "Output" "Result")
TASK_WORDS_NEGATIVE=("Wait" "<|endoftext|>" "Alternatively" "Check" "#" "%")
STEER_SUFFIXES_POSITIVE=("20" "21")
STEER_SUFFIXES_NEGATIVE=("20" "10")

TASK_THREAD_ID="${CUDA_VISIBLE_DEVICES}"

CACHE_ARG=()
if [ -n "${HF_CACHE_DIR}" ]; then
    CACHE_ARG=(--cache_dir "${HF_CACHE_DIR}")
fi

# -- helpers ------------------------------------------------------------------

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

# -- preprocess ---------------------------------------------------------------

echo "preprocess"
for task_name in "${TASK_NAMES[@]}"; do
    python3 preprocess.py \
        --input_file="${DATASET_DIR}/${task_name}/train.json" \
        --json_out_name="${task_name}.json" \
        --visualize_dir="${VISUALIZE_DIR}"
done

# -- run reflection levels ----------------------------------------------------

echo "run reflection levels"
for model_name in "${MODEL_NAMES[@]}"; do
    for task_word in "${TASK_WORDS_LEVELS[@]}"; do
        for task_name in "${TASK_NAMES[@]}"; do
            run_gt "$(get_limit "${task_name}")" "${task_name}" "${task_word}" \
                "${TASK_THREAD_ID}" "${model_name}" "step1"
        done
    done
done

# -- build steering vectors ---------------------------------------------------

echo "build steering vectors"
for i in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$i]}"
    hf_id="${HF_IDS[$i]}"
    if [ "${model_name}" = "${QWEN_MODEL_NAME}" ]; then
        wait_token_l1="${WAIT_TOKEN_L1_QWEN}"
    else
        wait_token_l1="${WAIT_TOKEN_L1_GEMMA}"
    fi

    for task_name in "${TASK_NAMES[@]}"; do
        input_file="${VISUALIZE_DIR}/${task_name}/step0/${task_name}.json"
        output_dir="${VISUALIZE_DIR}/${task_name}/${model_name}/step2"

        # L2-L0
        python3 build_vectors.py \
            --input_file="${input_file}" --model_name="${hf_id}" "${CACHE_ARG[@]}" \
            --output_dir="${output_dir}/steer_${S_LIMIT}_20" \
            --wait_token_1 ${WAIT_TOKEN_L2} --wait_token_2 ${WAIT_TOKEN_L0} \
            --limit "${S_LIMIT}"

        # L2-L1
        python3 build_vectors.py \
            --input_file="${input_file}" --model_name="${hf_id}" "${CACHE_ARG[@]}" \
            --output_dir="${output_dir}/steer_${S_LIMIT}_21" \
            --wait_token_1 ${WAIT_TOKEN_L2} --wait_token_2 ${wait_token_l1} \
            --limit "${S_LIMIT}"

        # L1-L0
        python3 build_vectors.py \
            --input_file="${input_file}" --model_name="${hf_id}" "${CACHE_ARG[@]}" \
            --output_dir="${output_dir}/steer_${S_LIMIT}_10" \
            --wait_token_1 ${wait_token_l1} --wait_token_2 ${WAIT_TOKEN_L0} \
            --limit "${S_LIMIT}"

        # embedding baseline
        python3 build_vectors.py \
            --input_file="${input_file}" --model_name="${hf_id}" "${CACHE_ARG[@]}" \
            --is_baseline=1 --output_dir="${output_dir}/steer_baseline" \
            --wait_token_1 ${WAIT_TOKEN_L2} --wait_token_2 "" \
            --limit "${S_LIMIT}"

        for steer_type in "steer_${S_LIMIT}_21" "steer_${S_LIMIT}_20" "steer_${S_LIMIT}_10" "steer_baseline"; do
            python3 reselect_words.py \
                --input_dir="${output_dir}/${steer_type}" \
                --word_limit "${WORD_LIMIT}"
        done
    done
done

# -- instruction selection ----------------------------------------------------

echo "instruction selection"
TASK_NAME_STEP3="gsm8k_adv"
for model_name in "${MODEL_NAMES[@]}"; do
    if [ "${model_name}" = "${QWEN_MODEL_NAME}" ]; then
        layers=("${LAYERS_QWEN[@]}")
    else
        layers=("${LAYERS_GEMMA[@]}")
    fi

    # embedding baseline (layer = -1)
    word_file="${VISUALIZE_DIR}/${TASK_NAME_STEP3}/${model_name}/step2/steer_baseline/word_-1.txt"
    if [ -f "${word_file}" ]; then
        while IFS= read -r task_word; do
            [ -z "${task_word}" ] && continue
            run_gt "${LIMIT_GSM8K}" "${TASK_NAME_STEP3}" "${task_word}" \
                "${TASK_THREAD_ID}" "${model_name}" "step3"
        done < "${word_file}"
    fi

    # layer rank, top 8 words per layer
    for steer_type in "steer_${S_LIMIT}_21" "steer_${S_LIMIT}_20"; do
        steer_dir="${VISUALIZE_DIR}/${TASK_NAME_STEP3}/${model_name}/step2/${steer_type}"
        for layer in "${layers[@]}"; do
            word_file="${steer_dir}/word_${layer}.txt"
            if [ ! -f "${word_file}" ]; then continue; fi
            j=0
            while IFS= read -r task_word; do
                [ -z "${task_word}" ] && continue
                if [ "${j}" -le 7 ]; then
                    run_gt "${LIMIT_GSM8K}" "${TASK_NAME_STEP3}" "${task_word}" \
                        "${TASK_THREAD_ID}" "${model_name}" "step3"
                    j=$((j + 1))
                fi
            done < "${word_file}"
        done
    done
done

# -- activation steering ------------------------------------------------------

echo "activation steering"
for model_name in "${MODEL_NAMES[@]}"; do
    if [ "${model_name}" = "${QWEN_MODEL_NAME}" ]; then
        layers=("${LAYERS_QWEN[@]}")
    else
        layers=("${LAYERS_GEMMA[@]}")
    fi

    for task_name in "${TASK_NAMES[@]}"; do
        limit=$(get_limit "${task_name}")

        for l in "${layers[@]}"; do
            for task_word in "${TASK_WORDS_POSITIVE[@]}"; do
                for s_suffix in "${STEER_SUFFIXES_POSITIVE[@]}"; do
                    steer_file="${VISUALIZE_DIR}/${task_name}/${model_name}/step2/steer_${S_LIMIT}_${s_suffix}/seed_avg.json"
                    run_steered "${limit}" "${task_name}" "${task_word}" "${TASK_THREAD_ID}" \
                        "${model_name}" "step4" "${l}" "1" "${steer_file}" "${s_suffix}"
                done
            done
        done

        for l in "${layers[@]}"; do
            for task_word in "${TASK_WORDS_NEGATIVE[@]}"; do
                for s_suffix in "${STEER_SUFFIXES_NEGATIVE[@]}"; do
                    steer_file="${VISUALIZE_DIR}/${task_name}/${model_name}/step2/steer_${S_LIMIT}_${s_suffix}/seed_avg.json"
                    run_steered "${limit}" "${task_name}" "${task_word}" "${TASK_THREAD_ID}" \
                        "${model_name}" "step4" "${l}" "-1" "${steer_file}" "${s_suffix}"
                done
            done
        done
    done
done

# -- plot results -------------------------------------------------------------

echo "plot results"
python3 plot.py \
    --visualize_dir "${VISUALIZE_DIR}" \
    --model_names   "${MODEL_NAMES[@]}" \
    --dataset_names "${TASK_NAMES[@]}" \
    --s_limit       "${S_LIMIT}" \
    --limits        "${LIMIT_GSM8K}" "${LIMIT_CRUXEVAL}"
