import os
import psutil
import multiprocessing
from functools import partial
import gc
import torch
import numpy as np
from tqdm import tqdm
from huggingface_hub import login

from datasets import load_from_disk, load_dataset, DatasetDict, concatenate_datasets
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

import evaluate

# LoRA
import bitsandbytes
from peft import prepare_model_for_kbit_training, prepare_model_for_int8_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR



# Dataset
def loadDataset(dataset_name: str):
    global num_cores
    
    dataset = load_dataset(dataset_name, num_proc=num_cores-12)
    print(f"{dataset}\nDataset lode complete\n")
    return dataset

def loadDatasetFromLocal(dataset_name: str):
    global num_cores
    
    dataset = load_from_disk(dataset_name)
    print(f"{dataset}\nDataset lode complete\n")
    return dataset

    
def splitDataset(dataset):
    # 데이터셋을 분할할 비율을 정합니다.
    train_ratio = 0.9  # 예를 들어 80%를 train set으로 사용하고 20%를 test set으로 사용하려면 0.8로 설정합니다.
    valid_ratio = 0.1
    # 데이터셋을 분할합니다.
    num_samples = len(dataset['train'])
    num_train_samples = int(train_ratio * num_samples)
    num_valid_samples = int(valid_ratio * num_samples)
    # train set과 test set을 생성합니다.
    train_set = dataset['train'].select(indices=range(num_train_samples))
    valid_set = dataset['train'].select(indices=range(num_train_samples, num_samples))
    # build dataset
    data = DatasetDict({
        'train': train_set,
        'validation': valid_set,
    })
    print(f"{data}\nDataset split complete!\n")
    return data

def extract_feature(batch, feature_extractor, tokenizer):
    audio = batch
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["audio"], sampling_rate=16000).input_features[0]
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids

    # cpu_percent = psutil.cpu_percent()
    # memory = psutil.virtual_memory()
    #print(f"CPU 사용량: {cpu_percent}%, 사용 가능한 메모리: {memory.available/1000000000}GB")
    return batch

@dataclass
class DatasetPrepper:
    processor: WhisperProcessor

    def __call__(self, batch):
        audio = batch

        batch["input_features"] = self.processor.feature_extractor(audio["audio"], sampling_rate=16000).input_features[0]

        batch["labels"] = self.processor.tokenizer(batch["text"]).input_ids
        return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# prepare datset
def prepare_dataset(dataset, feature_extractor, tokenizer, num_cores):
    partial_prepare_dataset = partial(extract_feature, feature_extractor=feature_extractor, tokenizer=tokenizer)
    dataset = dataset.map(partial_prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=num_cores-12)
    print(f"{dataset}\nDataset prepare complete!")
    return dataset

def compute_metrics(pred):
    global tokenizer
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

# load model
def load_model(model_checkpoint, repo_name):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_checkpoint)
    tokenizer = WhisperTokenizer.from_pretrained(model_checkpoint, language="ko", task="transcribe")    
    tokenizer = WhisperTokenizer.from_pretrained(model_checkpoint, language="ko", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_checkpoint, language="ko", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint, load_in_8bit=True, device_map="auto")
    print('Load Model Complete!')
    
    # model config
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.config.dropout = 0.1
    model.config.suppress_tokens = []
    # add to detection language
    model.config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="ko", task="transcribe")
    model.generation_config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="ko", task="transcribe")
    model.generation_config.suppress_tokens = []
    # UserWaarning 
    torch.utils.checkpoint.use_reentrant=True
    

    # training arguments
    training_args = Seq2SeqTrainingArguments(
        logging_dir=f"./checkPoints/{repo_name}",
        output_dir=f"./checkPoints/{repo_name}",  # change to a reapo name of your choice
        per_device_train_batch_size=8,         # ( int, 선택 사항 , 기본값은 8) – 훈련을 위한 GPU/TPU 코어/CPU당 배치 크기
        per_device_eval_batch_size=8,          #  ( int, 선택 사항 , 기본값은 8) – 평가를 위한 GPU/TPU 코어/CPU당 배치 크기
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        num_train_epochs=3,                 
        learning_rate=1e-3,             # ( float, 선택 사항 , 기본값은 5e-5) – Adam의 초기 학습률 
        warmup_steps=50,               #  ( int, 선택 사항 , 기본값은 0) – 0에서 까지 선형 준비에 사용되는 단계 수
        logging_steps=25,                #  ( int, 선택 사항 , 기본값은 500) – 두 로그 사이의 업데이트 단계 수
        eval_steps=500,                 # ( int, 선택 사항 ) – if 두 평가 사이의 업데이트 단계 수
        save_steps=500,                 # ( int, 선택 사항 , 기본값은 500) – 두 개의 체크포인트가 저장되기 전의 업데이트 단계 수
        gradient_checkpointing=True,
        generation_max_length=128,      #  max_length각 평가 루프에서 사용할 때,  predict_with_generate=True. 기본적으로 max_length모델 구성 값이 사용
        evaluation_strategy="epoch",    # ( str또는 IntervalStrategy , 선택 사항 , 기본값은 "no") — 훈련 중에 채택할 평가 전략 ["no", "steps", "epochs"]
        logging_strategy = "epoch",     #  ( str또는 IntervalStrategy , 선택 사항 , 기본값은 "steps") — 훈련 중에 채택할 로깅 전략 ["no", "steps", "epochs"]
        #predict_with_generate=True,     # 생성 메트릭(ROUGE, BLEU)을 계산하기 위해 생성을 사용할지 여부
        #metric_for_best_model="cer",    # 서로 다른 두 모델을 비교하는 데 사용할 측정항목을 지정하려면 와 함께 사용
        #greater_is_better=True,        # 더 나은 모델이 더 큰 메트릭을 가져야 하는지 여부
        #load_best_model_at_end=True,   # 훈련이 끝날 때 훈련 중에 찾은 최상의 모델을 로드할지 여부
        optim="adamw_bnb_8bit",        # default="adamw_hf", "adamw_torch", "adamw_8bit", "adamw_bnb_8bit",  
        fp16=False,                      # 최신 GPU에서 학습 속도를 높입니다.
        seed=42,
        report_to=["tensorboard"],     # 결과 및 로그를 보고할 통합 목록
        push_to_hub=True,              # 모델을 Hugging Face Hub에 푸시할지 여부,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    return training_args, model, processor, tokenizer, feature_extractor

def set_trainer(training_args, model, dataset, compute_metrics, processor, optimizer):
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    # optimizer = optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=training_args.max_steps, gamma=0.1)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        #compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback],  # LoRA
    )
    processor.save_pretrained(training_args.output_dir)
    
    return trainer

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control
    
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_cores = multiprocessing.cpu_count()

    # huggingface login
    HS_token = "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"
    login(token=HS_token)
    
    # load&split dataset
    dataset_name = "haseong8012/child-10k"
    dataset = loadDataset(dataset_name)
    dataset = splitDataset(dataset)

    # load model
    model_checkpoint = "openai/whisper-medium"
    special = "_LoRA"
    repo_name = f"{model_checkpoint.split('/')[1]}_{dataset_name.split('/')[-1]}{special}" 
    # repo_name = f"{model_checkpoint.split('/')[1]}_child-50k"  
    training_args, model, processor, tokenizer, feature_extractor = load_model(model_checkpoint, repo_name)

    # prepare dataset
    dataset = prepare_dataset(dataset, feature_extractor, tokenizer, num_cores)
    
    
    #Post-processing on the model
    # model = prepare_model_for_kbit_training(model)
    model = prepare_model_for_int8_training(model)
    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
    
    # Apply Low-rank adapters (LoRA) to the model
    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    # set trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    trainer = set_trainer(training_args, model, dataset, compute_metrics, processor, optimizer)
    
    trainer.train()
    

    # hub_path = "haseong8012"
    # model_id = f"{model_checkpoint.split('/')[-1]}_{dataset_name.split('/')[-1]}"
    # special = "_RoRA"   
    # peft_model_id = f"{hub_path}/{model_id}{special}"
    # model.push_to_hub(peft_model_id)