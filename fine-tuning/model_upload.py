import psutil
import multiprocessing
from functools import partial
import gc
import torch
from huggingface_hub import login

from datasets import load_from_disk, load_dataset, Dataset, DatasetDict, concatenate_datasets 
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

import evaluate

def loadDatasetLocal(dataset: str):
    dataset = load_from_disk(dataset_path + dataset_name)
    print(f"{dataset}\nDataset lode complete\n")
    return dataset

def loadDataset(dataset_name: str):
    dataset = DatasetDict()
    dataset = load_dataset(dataset_name)
    print(f"{dataset}\nDataset lode complete\n")
    return dataset

def splitDataset(dataset):
    # 데이터셋을 분할할 비율을 정합니다.
    train_ratio = 0.8  # 예를 들어 80%를 train set으로 사용하고 20%를 test set으로 사용하려면 0.8로 설정합니다.
    valid_ratio = 0.1
    test_ratio = 0.1
    # 데이터셋을 분할합니다.
    num_samples = len(dataset['train'])
    num_train_samples = int(train_ratio * num_samples)
    num_valid_samples = int(valid_ratio * num_samples)
    num_test_samples = int(test_ratio * num_samples)
    # train set과 test set을 생성합니다.
    train_set = dataset['train'].select(indices=range(num_train_samples))
    valid_set = dataset['train'].select(indices=range(num_train_samples, num_train_samples+num_valid_samples))
    test_set = dataset['train'].select(indices=range(num_train_samples+num_valid_samples, num_samples))
    # build dataset
    data = DatasetDict({
        'train': train_set,
        'validation': valid_set,
        'test': test_set
    })
    print(f"{data}\nDataset split complete!\n")
    return data

@dataclass
class DatasetPrepper:
    processor: WhisperProcessor

    def __call__(self, batch):
        audio = batch

        batch["input_features"] = self.processor.feature_extractor(audio["audio"], sampling_rate=16000).input_features[0]

        batch["labels"] = self.processor.tokenizer(batch["text"]).input_ids
        return batch
    

def prepare_dataset(batch, feature_extractor, tokenizer):
    audio = batch
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["audio"], sampling_rate=16000).input_features[0]
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids

    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    #print(f"CPU 사용량: {cpu_percent}%, 사용 가능한 메모리: {memory.available/1000000000}GB")
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
    



if __name__ == "__main__":
    # Hugging face hub login
    JW_token = "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"
    HS_token = "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"
    login(token=HS_token)

    # load dataset
    # ## from local
    # dataset_path = "D:\\stt\\data\\encoded_dataset_resampling\\"
    # dataset_name = "korean-child-command-voice_train-0-10000"
    # dataset = loadDatasetLocal(dataset_path+dataset_name)
    # from hugging face hub
    dataset_name = "haseong8012/child-50k"      # dataset_name 지정!!
    dataset = loadDataset(dataset_name)
    # split child  dataset
    dataset = splitDataset(dataset)
    print('Split Dataset Complete!')
    print(dataset)
    

    # # split dataset
    # data = splitDataset(dataset)

    # load model
    model_checkpoint = "whisper-tiny_child-50k"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_checkpoint)
    tokenizer = WhisperTokenizer.from_pretrained(model_checkpoint)
    processor = WhisperProcessor.from_pretrained(model_checkpoint)
    model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # model config
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.config.dropout = 0.1
    model.config.suppress_tokens = []
    # add to detection language
    model.config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="ko", task="transcribe")
    model.generation_config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="ko", task="transcribe")
    model.generation_config.suppress_tokens = []

    # training arguments
    repo_name = f"{model_checkpoint}"        # repp_name 지정!!
    # repo_name = "whisper-tiny_child-10k-adult-6k"        # repp_name 지정!!
    training_args = Seq2SeqTrainingArguments(
        logging_dir=f"checkPoints/{repo_name}",
        output_dir=repo_name,  # change to a reapo name of your choice
        per_device_train_batch_size=32,         # ( int, 선택 사항 , 기본값은 8) – 훈련을 위한 GPU/TPU 코어/CPU당 배치 크기
        per_device_eval_batch_size=16,          #  ( int, 선택 사항 , 기본값은 8) – 평가를 위한 GPU/TPU 코어/CPU당 배치 크기
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=3.75e-5,             # ( float, 선택 사항 , 기본값은 5e-5) – Adam의 초기 학습률 
        warmup_steps=500,               #  ( int, 선택 사항 , 기본값은 0) – 0에서 까지 선형 준비에 사용되는 단계 수
        max_steps=5000,                 # ( int, 선택 사항 , 기본값은 -1) – 양수로 설정된 경우 수행할 총 훈련 단계 수
        logging_steps=25,                #  ( int, 선택 사항 , 기본값은 500) – 두 로그 사이의 업데이트 단계 수
        eval_steps=500,                 # ( int, 선택 사항 ) – if 두 평가 사이의 업데이트 단계 수
        save_steps=500,                 # ( int, 선택 사항 , 기본값은 500) – 두 개의 체크포인트가 저장되기 전의 업데이트 단계 수
        gradient_checkpointing=True,
        generation_max_length=225,      #  max_length각 평가 루프에서 사용할 때,  predict_with_generate=True. 기본적으로 max_length모델 구성 값이 사용
        evaluation_strategy="steps",    # ( str또는 IntervalStrategy , 선택 사항 , 기본값은 "no") — 훈련 중에 채택할 평가 전략 ["no", "steps", "epochs"]
        logging_strategy = "steps",     #  ( str또는 IntervalStrategy , 선택 사항 , 기본값은 "steps") — 훈련 중에 채택할 로깅 전략 ["no", "steps", "epochs"]
        predict_with_generate=True,     # 생성 메트릭(ROUGE, BLEU)을 계산하기 위해 생성을 사용할지 여부
        #metric_for_best_model="cer",    # 서로 다른 두 모델을 비교하는 데 사용할 측정항목을 지정하려면 와 함께 사용
        #greater_is_better=True,        # 더 나은 모델이 더 큰 메트릭을 가져야 하는지 여부
        #load_best_model_at_end=True,   # 훈련이 끝날 때 훈련 중에 찾은 최상의 모델을 로드할지 여부
        optim="adamw_torch",        # default="adamw_hf", "adamw_torch" 
        fp16=True,                      # 최신 GPU에서 학습 속도를 높입니다.
        seed=42,
        report_to=["tensorboard"],     # 결과 및 로그를 보고할 통합 목록
        push_to_hub=True
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    processor.save_pretrained(training_args.output_dir)

    # test
    # prediction = trainer.predict(data["test"])
    # print("Test result: {prediction.metrics}")
    # # 테스트 결과를 파일로 저장합니다.
    # with open(f"{repo_name}/test_results.txt", "w") as f:
    #     f.write(str(prediction.metrics))

    # save model
    kwargs = {
        "dataset_tags": dataset_name,
        "dataset": dataset_name,  # a 'pretty' name for the training dataset
        "dataset_args": "config: ko, split: train, validation",
        "language": "ko",
        "model_name": repo_name,  # a 'pretty' name for our model
        "finetuned_from": "openai/whisper-tiny",
        "tasks": "automatic-speech-recognition",
        "tags": "hf-asr-leaderboard",
    }    
    trainer.push_to_hub(**kwargs)
    print("Model push to hub complete!")

    # clean up
    gc.collect
    torch.cuda.empty_cache()