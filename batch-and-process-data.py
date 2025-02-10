
# Imports

from typing import List, Dict, Union
from pydantic import PositiveInt
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import numpy as np

## Temporary variables for testing ########### (Remove)

TESTING_SUBSET_SIZE: PositiveInt = 15


# Configurables

BATCH_SIZE: PositiveInt = 5
DATASET_ID = "david-thrower/smol-smoltalk-plus-reasoning-synthetic-data"
TOKENIZER_ID: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


# Tokenizer:

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

special_tokens = ["<prompt>",
                  "</prompt>",
                  "<reason>",
                  "</reason>",
                  "<response>",
                  "</response>"]

special_tokens_dict = {'additional_special_tokens': special_tokens }
tokenizer.add_special_tokens(special_tokens_dict)



# Tokenizer constants

MAX_LENGTH = tokenizer.model_max_length
VOCABULARY_SIZE = tokenizer.vocab_size
TOKEN_TO_SPLIT_ON: PositiveInt = tokenizer("</prompt>")['input_ids'][0]
PAD_TOKEN: PositiveInt = tokenizer.pad_token_id


# Dataset
ds = load_dataset(DATASET_ID)

train_ds = ds["train"]
test_ds = ds["test"]

# Subset to validat what we are doing
batch_to_test = Dataset.from_dict(train_ds[:TESTING_SUBSET_SIZE])


def annotate_samples_with_special_tokens(data: dict) -> List[str]:
    texts = []
    for i in np.arange(len(data['prompt'])):
        text = f"<prompt>{data['prompt'][i]}</prompt>"\
                + f"<reason>{data['reason'][i]}</reason>"\
                + f"<response>{data['response'][i]}</response>"
        # print(f"text: {i}: {text}")
        texts.append(text)
    # print(texts)
    return texts


def new_tokenize_data(annotated_samples: List[str]) -> List[List[int]]:
        tokenized_data = []
        for i in np.arange(len(annotated_samples)):
            tokens = tokenizer(
                    annotated_samples[i], 
                    padding="max_length", 
                    max_length=MAX_LENGTH, 
                    padding_side="right",
                    truncation=True 
            )['input_ids']
            tokenized_data.append(tokens)
        return tokenized_data



def split_list(input_list: List[List[int]],
               token_to_split_on: PositiveInt) -> Dict:
    
    for i in np.arange(len(input_list)):
        token_index = input_list.index(token_to_split_on)
        # print(f"index: {token_index}")
        prompt = input_list[:token_index + 1] # + [token_to_split_on]
        # print(f"prompt: {prompt}")
        response = input_list[token_index + 1:]
        # print(f"response: {response}")
    return {'prompt': prompt, 'response': response}


def expand_tokens(data: List[List[int]],
                  token_to_split_on: PositiveInt,
                  pad_token: PositiveInt) -> Dict:
    for i in np.arange(len(data)):
        sample_0 = data[i]
        if token_to_split_on in sample_0:
            prompt_and_response = split_list(sample_0, token_to_split_on)
            prompt = prompt_and_response['prompt'] 
            response = prompt_and_response['response']
            new_data = []
            for i in np.arange(len(response)):
                if i != 0:
                    new_data.append(prompt + response[:i] + [pad_token for i in np.arange(len(response) - i)])
                else:
                    new_data.append(prompt + [pad_token for i in np.arange(len(response) - i)])
            return {'prompt': new_data, 'response': response}


def batch_data(data: Dataset,  batch_size: int):
    num_samples: PositiveInt  = len(data['prompt'])
    # print(f"Num samples: {num_samples}")
    completed_samples = []
    completed_labels = []
    for i in np.arange(start=0, step=batch_size, stop=num_samples):
        data_0 = data[i:i + batch_size] # -> dict
        annotated_data_0 = annotate_samples_with_special_tokens(data=data_0)
        # print(annotated_data_0[0])
        tokens_0 = new_tokenize_data(annotated_data_0)
        data_and_labels = expand_tokens(data=tokens_0,
                                        token_to_split_on=TOKEN_TO_SPLIT_ON,
                                        pad_token=PAD_TOKEN)
        if data_and_labels is not None:
            sample_0 = data_and_labels['prompt']
            label_0 = data_and_labels['response']
            for i in np.arange(len(sample_0)):
                completed_samples.append(sample_0[i])
                completed_labels.append(label_0[i])
    return {'data': completed_samples,
            'labels': completed_labels}


processed_tokens = batch_data(data=batch_to_test, batch_size=BATCH_SIZE)

lengths = np.array([len(processed_tokens['data'][i]) 
 for i in np.arange(len(processed_tokens['data']))])

print(f"Num expanded sanples: {lengths.shape} max: {lengths.max()}, min: {lengths.min()}")

for i in np.arange(20):
    sample = tokenizer.decode(processed_tokens['data'][i]).replace("<|im_end|>","")
    label = tokenizer.decode(processed_tokens['labels'][i]).replace("<|im_end|>","")
    print(f"Sample {i}: {sample}")
    print("\n")
    print(f"Label {i}: {label}")


import pandas as pd
processed_tokens_df = pd.DataFrame(processed_tokens)
processed_tokens_df.to_csv('2025-09-09--processed-tokens.csv')

## To Do: Add a dataset featuring an Arrow Streaming Writer 

