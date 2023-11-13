---
language:
- ko
license: apache-2.0
base_model: openai/whisper-tiny
tags:
- hf-asr-leaderboard
- generated_from_trainer
datasets:
- haseong8012/child-10k
model-index:
- name: whisper-tiny_child-10k_time-stretch
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# whisper-tiny_child-10k_time-stretch

This model is a fine-tuned version of [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) on the child-10k dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3.75e-05
- train_batch_size: 32
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 5000

### Framework versions

- Transformers 4.34.0
- Pytorch 2.1.0+cu121
- Datasets 2.14.5
- Tokenizers 0.14.1
