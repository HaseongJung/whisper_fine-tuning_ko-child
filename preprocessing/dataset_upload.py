import os
from dotenv import load_dotenv
from datasets import load_from_disk, DatasetDict, Dataset, load_dataset
from huggingface_hub import login
from huggingface_hub import HfApi

load_dotenv()
huggingFace_token = os.environ.get('huggingFace_token')

def uploadDataset(dataset: str, repo_name: str):
    loaded_dataset = load_from_disk(dataset)
    # loaded_encoded_dataset = load_dataset(dataset)
    print(f'Dataset load complete!:\n{loaded_dataset}')

    repo = repo_name
    login(token=huggingFace_token)
    loaded_dataset.push_to_hub(repo_name)
    print("Dataset upload complete!")

if __name__ == "__main__":
    dataset = "D:\\stt\\data\\encoded_dataset_resampling\\korean-child-command-voice_train-0-2070000"
    repo_name = "child-2.07M"
    uploadDataset(dataset, repo_name=repo_name)
    