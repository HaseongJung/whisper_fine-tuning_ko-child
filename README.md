# Whisper LoRA fine-tunning for Ko-child
![아이북 포스터 최종본](https://github.com/HaseongJung/whisper_fine-tuning_ko-child/assets/107913513/a0546cab-bc48-46cb-8233-5ee6e0f5570b)

<p align="justify">
아동들의 발음이 불완전할 수 있는 점을 고려하여, 소아와 유아의 음성에 특화된 모델을 개발하기 위해 Huggingface Hub에 공개된 OpenAI의 Whisper large-v2 모델을 미세조정(fine tuning)하였다.<br>
소아와 유아의 음성 데이터를 학습시키면서 기존에 학습되어 있던 일반 남녀의 음성 데이터에 대한 지식이 손상되는 것을 방지하기 위해 LoRA(Low-Rank Adaptation) fine tuning 기법을 적용하였다.<br>
<br>
연구를 위해 사용된 데이터셋은 AI-Hub에서 제공하고 있는 ‘명령어 음성(소아,유아)’ 데이터와 ‘명령어 음성(일반남여)’데이터의 일부를 가공하여 만들었다. ‘명령어 음성(소아,유아)’ 데이터에서 학습 데이터와 검증 데이터의 비율을 9:1(9000문장:1000문장)으로 설정하여 학습을 진행하였으며 테스트 데이터로 ‘명령어 음성(소아,유아)’ 10000 문장과 ‘명령어 음성(일반남여)’ 데이터 10000 문장을 사용하였다.<br>
<br>
모델의 성능은 각 테스트 데이터셋에 대한 CER(Character Error Rate) 값을 평가지표로 설정해 측정하였으며 표1에서와 같이 소아/유아 데이터와 일반 남녀 데이터에서 모두 성능이 향상되었음을 확인할 수 있다.<br>
<br>
최종적으로 서비스에 사용된 음성 인식 모델 파일과 관련 정보는 <a href="https://huggingface.co/haseong8012">Huggingface Hub</a>에서 확인할 수 있다.
</p>

<br>
#### 표 Whisper 모델 fine-tune 전후 성능 비교
| Model                      |	소아/유아(CER)	| 일반남여(CER) |
| :-----:                    | :-----:          |  :------:     |
| Whisper large-v2(original) |	   11.9	      |     12.8     |
| Whisper large-v2(ours)	   |     1.08	      |     10.6     |

