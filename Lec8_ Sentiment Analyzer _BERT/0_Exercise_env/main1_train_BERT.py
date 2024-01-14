import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from datasets import load_dataset
import numpy as np

from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer

from pprint import pprint

from datasets import load_metric

from torchinfo import summary

# -----------------------------------------------------------------------------

# Get and print the current working directory
current_working_directory = os.getcwd()
print(f"The current working directory is: {current_working_directory}")

# Change the working directory to a new directory (replace with the path you want)
new_working_directory = "C:/(...)/Transformers - BERT/src"
os.chdir(new_working_directory)

# Get and print the new current working directory
new_current_working_directory = os.getcwd()
print(f"The new current working directory is: {new_current_working_directory}")

# -----------------------------------------------------------------------------

def tokenize_fn(batch):
  return tokenizer(batch['sentence'], truncation=True)

def compute_metrics(logits_and_labels):
  # metric = load_metric("glue", "sst2")
  logits, labels = logits_and_labels
  predictions = np.argmax(logits, axis=-1)
  return metric.compute(predictions=predictions, references=labels)

# -----------------------------------------------------------------------------


raw_datasets = load_dataset("glue", "sst2")

raw_datasets
raw_datasets['train']
dir(raw_datasets['train'])
type(raw_datasets['train'])


raw_datasets['train'].data
raw_datasets['train'][0]
raw_datasets['train'][50000:50003]

raw_datasets['train'].features

# -----------------------------------------------------------------------------

# checkpoint = "bert-base-uncased"
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenized_sentences = tokenizer(raw_datasets['train'][0:3]['sentence'])

pprint(tokenized_sentences)


tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)

training_args = TrainingArguments(
  'my_trainer',
  evaluation_strategy='epoch',
  save_strategy='epoch',
  num_train_epochs=1,
)


model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2)


type(model)
model

summary(model)

# -----------------------------------------------------------------------------

params_before = []
for name, p in model.named_parameters():
  params_before.append(p.detach().cpu().numpy())



metric = load_metric("glue", "sst2")

metric.compute(predictions=[1, 0, 1], references=[1, 0, 0])

# -----------------------------------------------------------------------------
# Train and save BERT model

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


trainer.save_model('my_saved_model')


