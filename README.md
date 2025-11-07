# Implementation of: Unveiling the Latent Directions of Reflection in Large Language Models

[https://arxiv.org/abs/2508.16989](https://arxiv.org/abs/2508.16989)


## Installation

Install `lm_eval` from https://github.com/EleutherAI/lm-evaluation-harness

Download `Qwen2.5-3B` from https://huggingface.co/Qwen/Qwen2.5-3B, and put the files except `config.json` into `mymodel/MyQwen2.5-3B`

## Set up venv

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

## Run

```sh
bash 0_convert_sample.sh # Pre-processing
bash 1_run_levels.sh # Run Levels of Reflection
bash 2_build_steering_vec.sh # Build Steering Vectors
bash 3_run_instruction_rank.sh # Run the experiment of Instruction Selection by Steering Vector
bash 3_run_instruction_baseline.sh # Run the baseline experiment of Instruction Selection by Input Embeddings 
bash 4_run_steering_positive.sh # Run the experiment of Interference torward Enhancing Reflection
bash 4_run_steering_negative.sh # Run the experiment of Interference toward Inhibiting Reflection
python 5_plot_exp3_result.py # Plot the experiment result of step3
python 6_plot_exp4_result.py # Plot the experiment result of step4
```
