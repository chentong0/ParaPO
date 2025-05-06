# ParaPO: Aligning Language Models to Reduce Verbatim Reproduction of Pre-training Data

[Paper](https://arxiv.org/abs/2504.14452) | [Model checkpoints](https://huggingface.co/chentong00/Llama-3.1-8B-ParaPO) | [Data](https://huggingface.co/datasets/chentong00/ParaPO)

### Table of Contents

1. Overview  
2. Release of Checkpoints and Data  
3. Evaluation  
4. Customizing Dataset  
5. Model Training

---

## Overview

Language models often memorize parts of their pre-training data and reproduce them verbatim in open-ended tasks. This can raise concerns about copyright, plagiarism, or privacy. We introduce ParaPO, a preference optimization method that teaches models to distinguish between memorized and paraphrased text, and to avoid reproducing verbatim memorized content during inference.

Our results show that ParaPO enables models to avoid unintended regurgitation while preserving useful memorization (e.g., famous quotations). The use of online-collected data (i.e., sequences that the model has memorized) is key to effective reduction of reproduction, as discussed in our [paper](https://arxiv.org/abs/2504.14452).

This repository includes everything needed to replicate our results or apply ParaPO to other models:
- `libs`: source code of external packages
- `scripts`: main logic for data preparation, training, and evaluation
- `data`: evaluation datasets

---

## Release of Checkpoints and Data

We release all training data used in the study on [Huggingface](https://huggingface.co/datasets/chentong00/ParaPO). Each dataset includes `chosen` and `rejected` columns, formatted either as raw strings or structured message lists (chat format).

Available splits:
1. `parapo.pilecc.llama31.8b`: paraphrased vs. memorized sequences for Llama3.1-8B 
2. `parapo.pilecc.qwen25.7b`: paraphrased vs. memorized sequences for Qwen2.5-7B 
3. `parapo.pilecc.random`: Uniformly sampled Pile-CC paraphrased vs. original sequences  
4. `parapo.system.pilecc.llama31.8b`: System-prompted variant of (1)  
5. `generic.tulu3`: 16k examples sampled from [Tulu-3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)  
6. `generic.system.tulu3`: System-prompted variant of (5)

We also release ParaPO-tuned model checkpoints:
1. [Llama-3.1-8B-ParaPO](https://huggingface.co/chentong00/Llama-3.1-8B-ParaPO): Llama-3.1-8B trained on `parapo.pilecc.llama31.8b`.
2. [Llama-3.1-Tulu-3-8B-ParaPO](https://huggingface.co/chentong00/Llama-3.1-Tulu-3-8B-ParaPO): Llama-3.1-Tulu-3-8B tuned on `parapo.pilecc.llama31.8b`.
3. [Llama-3.1-Tulu-3-8B-ParaPO-System](https://huggingface.co/chentong00/Llama-3.1-Tulu-3-8B-ParaPO-System): Llama-3.1-Tulu-3-8B tuned on `parapo.system.pilecc.llama31.8b`.
4. [Llama-3.1-Tulu-3-8B-ParaPO-System-Mixing](https://huggingface.co/chentong00/Llama-3.1-Tulu-3-8B-ParaPO-System-Mixing): Llama-3.1-Tulu-3-8B tuned on a mixture of `parapo.system.pilecc.llama31.8b` and `generic.system.tulu3`.

---

## Evaluation

We evaluate both **regurgitation reduction** and **utility retention**. 

### Regurgitation Evaluation

All regurgitation evaluations use `eval_inference.py`. Run the following to evaluate regurgitation from various sources and prompts:

**Web Extraction**
```bash
python eval_inference.py --dataset extract --model_name ${MODEL_NAME} --model_tag ${MODEL_TAG} --prompt_inst completion --enable_chat
````

**Book Extraction**

```bash
python eval_inference.py --dataset booksum --model_name ${MODEL_NAME} --model_tag ${MODEL_TAG} --prompt_inst completion --enable_chat
```

**Creativity Index**

```bash
python eval_inference.py --dataset ci --model_name ${MODEL_NAME} --model_tag ${MODEL_TAG} --top_p 0.9 --temperature 0.7 --enable_chat
```

### Utility Evaluation

We assess retained utility via quotation recall and general benchmarks.

**Quotation Score**

```bash
python eval_inference.py --dataset shield --model_name ${MODEL_NAME} --model_tag ${MODEL_TAG} --top_p 0.9 --temperature 0.7 --enable_chat
```

For standard benchmarks (MMLU, GSM8K, BBH, IFEval, AlpacaEval2), we use the `open-instruct` package. Run the following inside `libs/open-instruct`:

**MMLU**

```bash
python -m eval.mmlu.run_eval --ntrain 0 --data_dir ${DATA_PATH}/mmlu/ --save_dir ${OUTPUT_PATH} --model_name_or_path ${MODEL_NAME} --tokenizer_name_or_path ${MODEL_NAME} --eval_batch_size 4 --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template
```

**GSM8K**

```bash
python -m eval.gsm.run_eval --data_dir ${DATA_PATH}/gsm/ --max_num_examples 200 --save_dir ${OUTPUT_PATH} --use_vllm --model_name_or_path ${MODEL_NAME} --tokenizer_name_or_path ${MODEL_NAME} --n_shot 8 --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template
```

**BBH**

```bash
python -m eval.bbh.run_eval --data_dir ${DATA_PATH}/bbh --save_dir ${OUTPUT_PATH} --model_name_or_path ${MODEL_NAME} --tokenizer_name_or_path ${MODEL_NAME} --max_num_examples_per_task 40 --use_vllm --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template
```

**IFEval**

```bash
python -m eval.ifeval.run_eval --data_dir ${DATA_PATH}/ifeval/ --save_dir ${OUTPUT_PATH} --model_name_or_path ${MODEL_NAME} --tokenizer_name_or_path ${MODEL_NAME} --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template --use_vllm
```

**AlpacaEval2**

```bash
python -m eval.alpaca_farm.run_eval --model_name_or_path ${MODEL_NAME} --tokenizer_name_or_path ${MODEL_NAME} --save_dir ./outputs-open-instruct/alpacaeval2/${MODEL_TAG} --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template --use_vllm
```

---

## Customizing Training Data

To generate your own preference pairs:

1. Filter memorized sequences from a corpus:

```bash
scripts/data/sampled_from_pile_memorized.py
```

* Set `model_name`, `model_tag`, and optionally modify `num_samples_to_take`
* We use `monology/pile-uncopyrighted` as the corpus.

2. Generate paraphrases:
   Use `generate_paraphrases.ipynb` to create corresponding paraphrases and pair them with the original memorized sequences.

These pairs can then be used as input to preference optimization training.

---

## Model Training

ParaPO uses DPO algorithm for model training, with one change: no chat templates are added to paraphrases pairs. This preserves the pre-training style format and avoids assumptions about input and output.

We use the [open-instruct](https://github.com/open-instruct/open-instruct) implementation. From the `libs/open-instruct` directory, run:

```bash
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$((TOTAL_BATCH_SIZE / NUM_GPUS / BATCH_SIZE_PER_GPU))

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/dpo_tune.py \
    --model_name_or_path ${MODEL_NAME} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --use_flash_attn \
    --gradient_checkpointing \
    --use_slow_tokenizer \
    --dataset_mixer_str '{"dpo_data_pile_cc_llama.jsonl": 1.0, "tulu3_8b_dpo_data.jsonl": 1.0}' \
    --max_seq_length 1024 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
    --gradient_accumulation_steps ${GRADIENT_ACC_STEPS} \
    --dpo_beta 0.1 \
    --learning_rate 5e-7 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --with_tracking \
    --report_to wandb \
    --push_to_hub False \
    --try_launch_beaker_eval_jobs False \
    --logging_steps 1
```

---

## Citation

If you find this work helpful, please cite:

```bibtex
@misc{chen2025parapoaligninglanguagemodels,
      title={ParaPO: Aligning Language Models to Reduce Verbatim Reproduction of Pre-training Data}, 
      author={Tong Chen and Faeze Brahman and Jiacheng Liu and Niloofar Mireshghallah and Weijia Shi and Pang Wei Koh and Luke Zettlemoyer and Hannaneh Hajishirzi},
      year={2025},
      eprint={2504.14452},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.14452}, 
}
```

For questions, feel free to open an issue or contact the authors via email.
