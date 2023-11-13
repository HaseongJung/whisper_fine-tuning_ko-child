import os
from dotenv import load_dotenv
import datasets  as ds
from datasets import load_from_disk
from huggingface_hub import login
from huggingface_hub import HfApi

load_dotenv()
huggingFace_token = os.environ.get('huggingFace_token')

def uploadDataset(dataset: str, repo_name: str):
    loaded_encoded_dataset = load_from_disk(dataset)
    print(f'Dataset load complete!:\n{loaded_encoded_dataset}')

    login(token=huggingFace_token)
    loaded_encoded_dataset.push_to_hub(repo_name)
    print("Dataset upload complete!")

if __name__ == "__main__":
    dataset = "D:\\stt\\data\\encoded_dataset_resampling\\korean-child-command-voice_train-0-100000"
    repo_name = "child-100K"
    uploadDataset(dataset, repo_name)
    
