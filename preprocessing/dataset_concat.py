import os
from dotenv import load_dotenv
import datasets as ds
from huggingface_hub import login
from huggingface_hub import HfApi

load_dotenv()
huggingFace_token = os.environ.get('huggingFace_token')

def concatDataset(start, end, step):    
    num1 = start    # num1 = start
    num2 = num1+step
    concated_dataset = ds.load_from_disk(f"./data/encoded_dataset_resampling/korean-child-command-voice_train-{0}-{start}")
    print(f'Current dataset\n{concated_dataset}')
    while (True):
        # dataset load
        loaded_dataset = ds.load_from_disk(f"./data/encoded_dataset_resampling/korean-child-command-voice_train-{num1}-{num2}")
        # merge dataset
        concated_dataset = ds.DatasetDict({"train": ds.concatenate_datasets([concated_dataset["train"], loaded_dataset["train"]])})

        num1, num2 = num1+step, num2+step
        if num1 >= end:
            break
        print(f'Current dataset\n{concated_dataset}')
    print(f'Dataset concatenate complete!\n{concated_dataset}')
    return concated_dataset



if __name__ == "__main__":
    
    start, end, step = 90000, 100000, 10000 
    concated_dataset = concatDataset(start, end, step)
    
    # Save dataset to local
    save_dir_name = f'korean-child-command-voice_train-0-{end}'   # save_path        
    concated_dataset.save_to_disk(f"D:/stt/data/encoded_dataset_resampling/{save_dir_name}")  # save dataset

    
    # dataset upload to Hugging face
    repo = f"child-{0}-{end}"         # repo 이름 지정!
    login(token=huggingFace_token)
    concated_dataset.push_to_hub(repo)
    print("Dataset upload complete!")
