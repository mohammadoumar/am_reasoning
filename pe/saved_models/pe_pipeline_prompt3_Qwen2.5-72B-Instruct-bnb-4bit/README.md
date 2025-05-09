---
library_name: peft
license: other
base_model: unsloth/Qwen2.5-72B-Instruct-bnb-4bit
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: pe_pipeline_prompt3_Qwen2.5-72B-Instruct-bnb-4bit
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# pe_pipeline_prompt3_Qwen2.5-72B-Instruct-bnb-4bit

This model is a fine-tuned version of [unsloth/Qwen2.5-72B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-72B-Instruct-bnb-4bit) on the pe_pipeline dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 5
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.49.0
- Pytorch 2.6.0+cu124
- Datasets 3.3.2
- Tokenizers 0.21.0