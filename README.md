# Whisper LoRA fine-tunning for Ko-child
### 생성형AI기반의 동화책 서비스인 '아이북'을 개발하기 위해 음성인식 fine-tunning을 진행하였다.

<p align="justify">
아동들의 발음이 불완전할 수 있는 점을 고려하여, 소아와 유아의 음성에 특화된 모델을 개발하기 위해 Huggingface Hub에 공개된 OpenAI의 Whisper large-v2 모델을 미세조정(fine tuning)하였다.<br>
소아와 유아의 음성 데이터를 학습시키면서 기존에 학습되어 있던 일반 남녀의 음성 데이터에 대한 지식이 손상되는 것을 방지하기 위해 LoRA(Low-Rank Adaptation) fine tuning 기법을 적용하였다.<br>
<br>
연구를 위해 사용된 데이터셋은 AI-Hub에서 제공하고 있는 ‘명령어 음성(소아,유아)’ 데이터와 ‘명령어 음성(일반남여)’데이터의 일부를 가공하여 만들었다. ‘명령어 음성(소아,유아)’ 데이터에서 학습 데이터와 검증 데이터의 비율을 9:1(9000문장:1000문장)으로 설정하여 학습을 진행하였으며 테스트 데이터로 ‘명령어 음성(소아,유아)’ 10000 문장과 ‘명령어 음성(일반남여)’ 데이터 10000 문장을 사용하였다.<br>
<br>
모델의 성능은 각 테스트 데이터셋에 대한 CER(Character Error Rate) 값을 평가지표로 설정해 측정하였으며 표1에서와 같이 소아/유아 데이터와 일반 남녀 데이터에서 모두 성능이 향상되었음을 확인할 수 있다.<br>
<br>
최종적으로 서비스에 사용된 음성 인식 모델 파일과 관련 정보는 <a href="https://huggingface.co/haseong8012">Huggingface Hub</a>에서 확인할 수 있다.<br>
<br>
</p>

### 최종 모델: haseong8012/whisper-large-v2_child10K_LoRA <br>
#### Training procedure <br>
The following bitsandbytes quantization config was used during training: <br>
- quant_method: bitsandbytes
- load_in_8bit: True
- load_in_4bit: False
- llm_int8_threshold: 6.0
- llm_int8_skip_modules: None
- llm_int8_enable_fp32_cpu_offload: False
- llm_int8_has_fp16_weight: False
- bnb_4bit_quant_type: fp4
- bnb_4bit_use_double_quant: False
- bnb_4bit_compute_dtype: float32
#### Framework versions
- PEFT 0.5.0

학습 데이터셋: <a href="https://huggingface.co/datasets/haseong8012/child-10k">haseong8012/child-10k (유아음성 10000문장)</a><br>
평가 데이터셋: <a href="https://huggingface.co/datasets/haseong8012/general10k_for-test">haseong8012/general10k_for-test(일반음성 10000문장)</a>, <a href="https://huggingface.co/datasets/haseong8012/child-10k_for-test">haseong8012/child-10k_for-test(유아음성 10000문장)</a>



### 표 Whisper 모델 fine-tune 전후 성능 비교
| Model                      |	소아/유아(CER)	| 일반남여(CER) |
| :-----:                    | :-----:          |  :------:     |
| Whisper large-v2(original) |	   11.9	      |     12.8     |
| Whisper large-v2(ours)	   |     1.08	      |     10.6     |

