import os
import gc
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
import wave
import json
import numpy as np
import re
import tqdm
import time
import pickle
import librosa
import time

import parmap
import multiprocessing
from multiprocessing import freeze_support

from huggingface_hub import login
from huggingface_hub import HfApi

load_dotenv()
huggingFace_token = os.environ.get('huggingFace_token')

def remove_nested_parentheses(text):
    # 정규 표현식을 사용하여 괄호와 내용을 추출합니다.
    pattern = r"\((SP|FP|NO|SN):(.*?)\)"
    
    # 모든 중첩된 괄호를 제거합니다.
    while re.search(pattern, text): text = re.sub(pattern, '\\2', text)
    
    return text

# (A)/(B)일 때, B만을 가져와주는 함수
def bracket_filter(sentence):
    new_sentence = str()
    flag = False

    for ch in sentence:
        if ch == '(' and flag == False:
            flag = True
            continue
        if ch == '(' and flag == True:
            flag = False
            continue
        if ch != ')' and flag == False:
            new_sentence += ch
    return new_sentence

# 문자 단위로 특수 문자 및 노이즈 표기 필터링 함수
def special_filter(sentence):
    SENTENCE_MARK = ['.', '?', ',', '!']
    NOISE = ['o', 'n', 'u', 'b', '!']
    EXPECT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';']
    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            # o/, n/ 등 처리
            if idx+1 < len(sentence) and ch in NOISE and sentence[idx+1] == '/':
                continue
        if ch == '%':
            new_sentence += "퍼센트"
        elif ch == '#':
            new_sentence += '샾'
        elif ch not in EXPECT:
            new_sentence += ch
    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence

def func(paired_file):
    wave_file, text_file = paired_file
    # WAV 파일의 경로와 데이터를 저장할 리스트 초기화
    # WAV 파일을 읽어서 데이터셋에 추가
        
    # WAV 파일 열기
    with wave.open(wave_file, 'rb') as wav:
        # WAV 파일 데이터 읽기
        audio_bytes = wav.readframes(-1)
        audio_array = (np.frombuffer(audio_bytes, dtype="int16")).astype(np.float32)    
        
        
        sample_rate = wav.getframerate()
        # resampling 48000 -> 16000
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        
        # 정규화 (Z-score normalization)
        normalized_audio = (audio_array - np.mean(audio_array)) / np.std(audio_array)
        # 정규화 (min-max normalization)
        # normalized_audio = (audio_array - np.min(audio_array)) / (np.max(audio_array) - np.min(audio_array))
        

    # JSON 파일을 읽어서 데이터프레임으로 변환
    with open(text_file, 'r', encoding='UTF8') as file:
        data = json.load(file)
        transcription = data['전사정보']['LabelText']
        cleaned_text = remove_nested_parentheses(transcription).rstrip()
        cleaned_text = bracket_filter(cleaned_text)
        cleaned_text = special_filter(cleaned_text)

    return normalized_audio, cleaned_text
    #return normalized_audio, [cleaned_text, data['파일정보']['FileName']]

# 데이터셋 생성
def makeDataDict(x: list):
    global result
    
    audio = []
    text = []
    for i in range(len(x)):
        audio.append(x[i][0])
        # "Python int too large to convert to C long" error발생하는 값 찾기
        # import sys
        # if x[i][0] > sys.maxsize:
            # print(x[i][0])
            
        text.append(x[i][1])
    del result
    gc.collect()
    custom_dataset = Dataset.from_dict({
        'text': text,
        'audio': audio
})
    return custom_dataset

if __name__ == '__main__':
    # Train dataset
    # global variables for audio and text paired data
    core_count = multiprocessing.cpu_count()
    
    # audio files load
    audio_dir = "D:\\stt\\data\\raw_data\\자유대화 음성(소아, 유아)\\Validation"
    wav_files = []
    for root, directories, files in tqdm.tqdm(os.walk(audio_dir), desc="Wav files loading..."):
        for file in (files):  # tqdm을 사용하여 진행률 표시
            file_path = os.path.join(root, file)  # 파일 경로를 올바르게 조합
            wav_files.append(file_path)        

    # json files load
    label_dir = "D:\\stt\\data\\raw_data\\자유대화 음성(소아, 유아)\\Validation"
    json_files = []
    for (root, directories, files) in tqdm.tqdm(os.walk(label_dir), desc="Json files loading..."):
        for file in (files):    
            file_path = root + '/' + file
            json_files.append(file_path) 
    print("Files load complete!")
    #print(len(wav_files))  #2077658
       
    # 10000씩 하자!
    start, end = 0, 20000 #len(wav_files)   # start, end 지정
    step = 20000                # step 지정 (파일 몇개씩 preprocessing할건지)    
    num1 = start
    num2 = num1+step
    while (True):
        if num1 >= end:
            break
        elif num1 == 2070000:
            num2 = len(wav_files)
        elif num1 > end:
            break
            
        start = time.time()
        print(f"{'-'*60} Current dataset: [{num1}~{num2}] / {len(wav_files)} {'-'*60}")
        # sub part of wav and json files
        wav_files1, json_files1 = wav_files[num1:num2], json_files[num1:num2]  

        freeze_support()
        result = parmap.map(func, list(zip(wav_files1, json_files1)), pm_pbar={"desc": "Processing files"}, pm_processes=core_count)
        print("Preprocessing complete!")
        del wav_files1
        del json_files1
        gc.collect()

        train_dataset = makeDataDict(result)

        # # Valid dataset
        # # WAV and Json 파일이 있는 폴더들 경로 설정 
        # audio_dir = "data/명령어 음성(소아, 유아)/Validation/audio/"
        # wav_files = []
        # for (root, directories, files) in os.walk(audio_dir):
        #     for file in files:
        #         file_path = root + '/' + file
        #         wav_files.append(file_path) 
        # wav_files.sort()
                
        # label_dir = "data/명령어 음성(소아, 유아)/Validation/label/"
        # json_files = []
        # for (root, directories, files) in os.walk(label_dir):
        #     for file in files:
        #         file_path = root + '/' + file
        #         json_files.append(file_path)
        # json_files.sort()

        # #num = 10
        # # sub part of wav and json files
        # wav_files1, json_files1 = wav_files[:], json_files[:]
        # freeze_support()
        # result = parmap.map(func, list(zip(wav_files1, json_files1)), pm_pbar=True, pm_processes=core_count)

        # valid_dataset = makeDataDict(result)  
        # print(valid_dataset, '\n', "Valid dataset build complete!")

        dataset = DatasetDict()
        dataset["train"] = train_dataset
        #dataset["validation"] = valid_dataset

        print(f"Dataset building complete!")
        
        save_dir_name = f'korean-train-command-voice_valid-{num1}-{num2}'    # 저장할 이름
        dataset.save_to_disk(f"D:/stt/data/encoded_dataset_resampling/{save_dir_name}")
        print(f"Dataset saving complete!")
        
        # 시간 측정
        end2 = time.time()
        print(f"{end2 - start:.5f} sec")        
    
        num1, num2 = num1+step, num2+step
        
    # dataset upload to Hugging face
    repo = f"korean-child-command-voice_train-{num1}-{num2}"         # repo 이름 지정!
    login(token=huggingFace_token)
    dataset.push_to_hub(repo)
    print("Dataset upload complete!")
    
    # audio-text matching check
    import sounddevice as sd
    sd.play(dataset["train"]["audio"][9], 48000)
    print(dataset["train"]["text"][9])
    sd.wait()
    