#!/usr/bin/env python
# coding: utf-8

# In[1]:


import transformers
from transformers import (
    MT5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, MT5Tokenizer, MT5Config
)

import datasets
import pandas as pd
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
from datasets import load_metric
import gc
import datasets
import os
import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="7"
os.environ["WANDB_DISABLED"] = "true"
get_ipython().system('export CUDA_VISIBLE_DEVICES=7')
device, use_gpu = ("cuda:7", True) if torch.cuda.is_available() else ("cpu", False)
import json


# In[2]:


get_ipython().system('nvidia-smi')


# In[3]:


checkpoint = "/workspace/home/chieunq/viT5-large-intend/checkpoint-12500"
model = MT5ForConditionalGeneration.from_pretrained(checkpoint)
print('load model done')
tokenizer = MT5Tokenizer.from_pretrained(checkpoint)
print('load tokenizer done')


# In[4]:


import re
def format(tmp):
    tmp = re.sub('=', 'E', tmp)
    tmp = re.sub('<', 'S', tmp)
    tmp = re.sub('>', 'B', tmp)
    return tmp
def load_data():
    data = []
    with open("data/augument_gold.jsonl",encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if len(line) < 1:
                break
            line = json.loads(line)
            
            data.append(
                {
                    'question': line['input'],
                    'sql': format(line['output'])
                }
            )
    
    print(f'total size of data is {len(data)}')
    
    tdata = pd.DataFrame(data)
    tdata = tdata.reset_index()
    dataset = datasets.Dataset.from_pandas(tdata)

    # don't care about test_size. 
    train = dataset.train_test_split(
        train_size=241, test_size=1, seed=42
    )
    return train
data = load_data()


# In[5]:


train_data = data['train']
test_data = data['test']


# In[6]:


def format_dataset(example):
     return {'input': example['question'], 'target': example['sql']}


# In[7]:


train_data = train_data.map(format_dataset, remove_columns=train_data.column_names)
test_data = test_data.map(format_dataset, remove_columns=test_data.column_names)


# In[8]:


train_data


# In[9]:


def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input'], pad_to_max_length=True, max_length=128)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target'], pad_to_max_length=True, max_length=128)
    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids'],
        'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings
train_data = train_data.map(convert_to_features, batched=True, remove_columns=train_data.column_names)
test_data = test_data.map(convert_to_features, batched=True, remove_columns=test_data.column_names)

# columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']

# train_data.set_format(type='torch', columns=columns)
# test_data.set_format(type='torch', columns=columns)


# In[10]:


# arr = [259, 13081, 317, 1266, 267, 2862, 1262, 270, 387, 259, 296, 352, 1541, 262, 387, 259, 296, 366, 6121, 355, 1492, 259, 296, 977, 1262, 268, 301, 5971, 330, 1269, 266, 259, 296, 977, 10470, 300, 2591, 330, 1269, 266, 259, 296, 441, 28811, 3123, 270, 259, 263, 5537, 5730, 355, 1687, 259, 260, 259, 102385, 267, 10830, 796, 355, 1262, 268, 301, 5971, 317, 708, 262, 259, 270, 2627, 259, 1369, 690, 1313, 387, 1970, 291, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# tokenizer.decode(arr)


# In[11]:


from datasets import load_metric
rouge = load_metric("rouge1.py")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# In[12]:


data_collator = DataCollatorForSeq2Seq(tokenizer,model=model)
training_args = Seq2SeqTrainingArguments(
    output_dir="viT5-large-intend-continue-1",
    per_device_train_batch_size=1,
    num_train_epochs=4,
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    evaluation_strategy="steps",
    do_train=True,
    do_eval=True,
    logging_steps=241,
    save_strategy="steps",
    save_steps=241,
    eval_steps=241,
    overwrite_output_dir=True,
    save_total_limit=5,
    load_best_model_at_end=True,
    report_to=None,
    #fp16=True, 
)
trainer = Seq2SeqTrainer(
    model=model,
    data_collator = data_collator,
    tokenizer = tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data,
)


# In[13]:


trainer.train()


# In[14]:


import re
se = re.compile(re.escape('Số (3)'),re.IGNORECASE)
ok = se.search('Cầu thủ số (3) đã chơi ở bao nhiêu trường?')
if ok:
    print('ok')


# In[15]:


get_ipython().system('nvidia-smi')


# In[16]:


import transformers
print(transformers.__version__)


# In[ ]:




