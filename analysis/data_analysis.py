import os
import wave
import json
import numpy as np
import librosa

audio_dir = "../data/raw_data/명령어 음성(소아, 유아)/Training/audio/"
wav_files = []
for root, directories, files in os.walk(audio_dir):
    for file in (files):  # tqdm을 사용하여 진행률 표시
        file_path = os.path.join(root, file)  # 파일 경로를 올바르게 조합
        wav_files.append(file_path)
        

# 데이터셋의 오디오 파일 총 수
total_audio_files = len(wav_files)

audio_durations = []  # 오디오 길이를 저장할 리스트
for wav_file in wav_files[:1000]:
    audio, sample_rate = librosa.load(wav_file, sr=None)
    
# WAV 파일을 읽어서 데이터셋에 추가
for wav_file in wav_files[:1000]:
    with wave.open(wav_file, 'rb') as wav:
        # WAV 파일 데이터 읽기
        audio_bytes = wav.readframes(-1)
        audio_array = np.frombuffer(audio_bytes, dtype="int16")
        # 오디오 길이 및 무음 구간 길이 측정
        audio_duration = len(audio_array) / 16000  # 샘플링 레이트에 따라 조절
        print(audio_duration)
        audio_durations.append(audio_duration)
        
# 평균 오디오 길이 계산
mean_duration = np.mean(audio_durations)
if audio_durations:
    min_duration = np.min(audio_durations)
else:
    min_duration = None
if audio_durations:
    max_duration = np.max(audio_durations)
else:
    max_duration = None

# 결과 출력
print(f"총 오디오 파일 수: {total_audio_files}")
print(f"평균 오디오 길이: {mean_duration:.2f} 초")
print(f"최소 오디오 길이: {min_duration:.2f} 초")
print(f"최대 오디오 길이: {max_duration:.2f} 초")