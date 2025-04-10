# ***************** Fine-Tuning LLMs on CDCP dataset *********************** #

# ********** Libraries and GPU *************

import os
import ast
import sys
import json
import torch
import pickle
import subprocess

sys.path.append('../')

import pandas as pd

from pathlib import Path
from tqdm import tqdm
from llamafactory.chat import ChatModel # type: ignore
from llamafactory.extras.misc import torch_gc # type: ignore
from sklearn.metrics import classification_report
#from utils.post_processing import post_process

try:    
    assert torch.cuda.is_available() is True
    
except AssertionError:
    
    print("Please set up a GPU before using LLaMA Factory...")


# ************** PATH SETTINGS *************

CURRENT_DIR = Path.cwd()
CDCP_DIR = CURRENT_DIR.parent
DATASET_DIR = CDCP_DIR / "datasets"

#ERC_DIR = FT_DIR.parent
LLAMA_FACTORY_DIR = CDCP_DIR.parent / "LLaMA-Factory"

BASE_MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"
LOGGING_DIR = CDCP_DIR / "training_logs"
OUTPUT_DIR = CDCP_DIR / "saved_models" / f"""cdcp_pipeline_{BASE_MODEL.split("/")[1]}"""



# ****************** DATASET FILES ******************


# # *** TRAIN/TEST DATASET NAME/FILENAME *** #

train_dataset_name = f"""cdcp_pipeline_train.json"""
test_dataset_name = f"""cdcp_pipeline_test.json"""

train_dataset_file = DATASET_DIR / train_dataset_name
test_dataset_file = DATASET_DIR / test_dataset_name


# # *** TRAIN ARGS FILE PATH *** #

if not os.path.exists(os.path.join(CDCP_DIR, "model_args")):
    os.mkdir(os.path.join(CDCP_DIR, "model_args"))

train_file = CDCP_DIR / "model_args" / f"""{train_dataset_name.split(".")[0].split("train")[0]}{BASE_MODEL.split("/")[1]}.json"""
#print(train_file)

# *** UPDATE dataset_info.json file in LLaMA-Factory *** #

dataset_info_line =  {
  "file_name": f"{train_dataset_file}",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}

with open(os.path.join(LLAMA_FACTORY_DIR, "data/dataset_info.json"), "r") as jsonFile:
    data = json.load(jsonFile)

data["cdcp_pipeline"] = dataset_info_line

with open(os.path.join(LLAMA_FACTORY_DIR, "data/dataset_info.json"), "w") as jsonFile:
    json.dump(data, jsonFile)


# # ************************** TRAIN MODEL ******************************#

NB_EPOCHS = 5

args = dict(
    
  stage="sft",                           # do supervised fine-tuning
  do_train=True,

  model_name_or_path=BASE_MODEL,         # use bnb-4bit-quantized Llama-3-8B-Instruct model
  num_train_epochs=NB_EPOCHS,            # the epochs of training
  output_dir=str(OUTPUT_DIR),                 # the path to save LoRA adapters
  overwrite_output_dir=True,             # overrides existing output contents

  dataset="cdcp_pipeline",                      # dataset name
  template="llama3",                     # use llama3 prompt template
  #train_on_prompt=True,
  val_size=0.1,
  max_samples=10000,                       # use 500 examples in each dataset

  finetuning_type="lora",                # use LoRA adapters to save memory
  lora_target="all",                     # attach LoRA adapters to all linear layers
  per_device_train_batch_size=2,         # the batch size
  gradient_accumulation_steps=4,         # the gradient accumulation steps
  lr_scheduler_type="cosine",            # use cosine learning rate scheduler
  loraplus_lr_ratio=16.0,                # use LoRA+ algorithm with lambda=16.0
  #temperature=0.5,
  
  warmup_ratio=0.1,                      # use warmup scheduler    
  learning_rate=5e-5,                    # the learning rate
  max_grad_norm=1.0,                     # clip gradient norm to 1.0
  
  fp16=True,                             # use float16 mixed precision training
  quantization_bit=4,                    # use 4-bit QLoRA  
  #use_liger_kernel=True,
  #quantization_device_map="auto",
  
  logging_steps=10,                      # log every 10 steps
  save_steps=5000,                       # save checkpoint every 1000 steps    
  logging_dir=str(LOGGING_DIR),
  
  # use_unsloth=True,
  report_to="none"                       # discards wandb

)

json.dump(args, open(train_file, "w", encoding="utf-8"), indent=2)

p = subprocess.Popen(["llamafactory-cli", "train", train_file], cwd=LLAMA_FACTORY_DIR)
p.wait()


# ********************** INFERENCES ON FINE_TUNED MODEL ******************** #

# LOAD MODEL, ADD LORA ADAPTERS #

args = dict(
    
  model_name_or_path=BASE_MODEL, # use bnb-4bit-quantized Llama-3-8B-Instruct model
  adapter_name_or_path=str(OUTPUT_DIR),            # load the saved LoRA adapters  
  template="llama3",                     # same to the one in training
  
  finetuning_type="lora",                  # same to the one in training
  quantization_bit=4,                    # load 4-bit quantized model
  #device_map="auto"
)

# if torch.cuda.is_available():
    
#     device = torch.device("cuda")
    
model = ChatModel(args)#.to(device) # type: ignore

# # LOAD TEST SET #

with open(test_dataset_file, "r+") as fh:
    test_dataset = json.load(fh)

test_prompts = []
test_grounds = []

for sample in test_dataset:
    test_prompts.append(sample["instruction"] + sample["input"])
    test_grounds.append(sample["output"])


# INFERENCE ON TEST SET #

test_predictions = []

for prompt in tqdm(test_prompts):

    messages = []
    messages.append({"role": "user", "content": prompt})

    response = ""
    
    for new_text in model.stream_chat(messages):
        #print(new_text, end="", flush=True)
        response += new_text
        #print()
    test_predictions.append(response)

    torch_gc()
    


# # SAVE GROUNDS AND PREDICTIONS *

results_d = {"grounds": test_grounds,
             "predictions": test_predictions}

with open(os.path.join(OUTPUT_DIR, f"""cdcp_pipeline_results_{NB_EPOCHS}.pickle"""), 'wb') as fh:

    pickle.dump(results_d, fh)


# **************************** POST-PROCESSING ************************ #