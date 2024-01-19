import os
import ast
import time
import torch
import wandb
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup

print("Pytorch Lightning", pl.__version__)

wandb.login()
COLUMNS = ['query', 'answer', 'ground_knowledge', 'ground_persona']
SPECIAL_TOKENS = [
    # "<machine>", "<human>",
    "<persona>", "<knowledge>", "<query>"]
ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': [
    # '<machine>', '<human>',
    '<persona>', '<knowledge>', '<query>']}
SPECIAL_TOKENS_MAP = {
    # "machine": "<machine>",
    # "human": "<human>",
    "persona": "<persona>",
    "knowledge": "<knowledge>",
    "query": "<query>"
}

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer)
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    num_added_tokens = len(SPECIAL_TOKENS)
    print("orig num", orig_num_tokens, "num_added", num_added_tokens)
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def ast_literal(example):
    return " </s> ".join(ast.literal_eval(example))

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
print("T5 Model and Tokenizer loaded and ready!")

print(f"Model Vocab Size: {model.config.vocab_size}",
          f"Tokenizer Vocab Size: {tokenizer.vocab_size}")
add_special_tokens_(model, tokenizer)
print(f"Model Vocab Size: {model.config.vocab_size}, Tokenizer Vocab Size: {tokenizer.vocab_size}")


val = "../../data/focus_val_data.csv"
train = "../../data/focus_train_data.csv"
test = "../../data/focus_test_data.csv"



# Load the training data and split it into training and validation sets
df_val = pd.read_csv(val)
df_train = pd.read_csv(train)
df_test = pd.read_csv(test)

df_val['ground_persona'] = df_val['ground_persona'].apply(ast_literal)
df_train['ground_persona'] = df_train['ground_persona'].apply(ast_literal)
df_test['ground_persona'] = df_test['ground_persona'].apply(ast_literal)


class DialogDataset(Dataset):
    def __init__(self, df, tokenizer, max_source_length = 1024,
                 max_target_length = 256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
      row = self.df.iloc[item]
      query = SPECIAL_TOKENS_MAP["query"] +" "+ str(row["query"])
      persona = SPECIAL_TOKENS_MAP["persona"] + " "+ str(row['ground_persona'])
      knowledge = SPECIAL_TOKENS_MAP["knowledge"] + " " + str(row["ground_knowledge"])
      answer = row['answer']

      text =  query + " " + persona + " " + knowledge
      
      source_encoding = tokenizer.encode_plus(text, 
                                    max_length=self.max_source_length,
                                    padding = 'max_length',
                                    return_attention_mask = True,
                                    add_special_tokens=True,
                                    truncation=True,
                                    return_tensors="pt"
                                    )
      target_encoding = tokenizer.encode_plus(answer, 
                                    max_length=self.max_target_length,
                                    padding = 'max_length',
                                    # return_attention_mask = True,
                                    add_special_tokens=True,
                                    truncation=True,
                                    return_tensors="pt"
                                    ).input_ids

      input_ids = source_encoding["input_ids"].flatten()
      attention_mask = source_encoding["attention_mask"].flatten()

      labels = target_encoding
      
      labels[labels == 0] = -100

      # labels_with_ignore_index = []
      # for labels_example in labels:
      #   labels_example = [label if label != 0 else -100 for label in labels_example]
      #   labels_with_ignore_index.append(labels_example)


      target_ids = labels.flatten()
      return {
          # "query": query,
          # "persona": persona,
          # "knowledge": knowledge,
          # "answer": answer,
          "input_ids":input_ids,
          "attention_mask":attention_mask,
          "labels": target_ids
          }


train_dataset = DialogDataset(df_train, tokenizer)
train_dataloader = DataLoader(train_dataset,  batch_size=8, num_workers=4)

valid_dataset = DialogDataset(df_val, tokenizer)
valid_dataloader = DataLoader(valid_dataset,  batch_size=8, num_workers=4)

test_dataset = DialogDataset(df_test, tokenizer)
test_dataloader = DataLoader(test_dataset,  batch_size=8, num_workers=4)


class CodeT5(pl.LightningModule):
    def __init__(self,  lr=5e-5, batch_size = 4, num_train_epochs=15, warmup_steps=1000):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.model.resize_token_embeddings(new_num_tokens=32103)
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):     
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return outputs
      
    def training_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", outputs.loss)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx)     
        self.log("validation_loss", outputs.loss,
                #  on_epoch=True
                 )

        return outputs.loss

    def test_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx)     

        return outputs.loss

    def configure_optimizers(self):
        # create optimizer
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # create learning rate scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * len(train_dataloader)
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return valid_dataloader

    def test_dataloader(self):
        return test_dataloader


model = CodeT5()



# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
early_stop_callback = EarlyStopping(
    monitor='validation_loss',
    patience=8,
    strict=False,
    verbose=True,
    mode='min'
)
lr_monitor = LearningRateMonitor(logging_interval='step')
wandb_logger = WandbLogger(name='codet5-finetune-personalized-answer-generation', project='CodeT5')
best_checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    auto_insert_metric_name= True,
    monitor="validation_loss",
    mode="min",
    dirpath="./CodeT5/ModelCheckpoint",
    filename="t5model-nohistory-best",
)

latest_checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    every_n_train_steps = 300,
    auto_insert_metric_name= True,
    monitor="training_loss",
    mode="min",
    dirpath="./CodeT5/ModelCheckpoint",
    filename="t5model-nohistory-latest",
)

trainer = Trainer(
    # fast_dev_run=True,
    auto_lr_find=True,
    # limit_train_batches = 0.3,
    # limit_val_batches = 0.3,
    auto_scale_batch_size="binsearch",
    gpus=1, 
    default_root_dir="./CodeT5/Checkpoints", 
    logger=wandb_logger, 
    callbacks=[early_stop_callback,
               lr_monitor,
               latest_checkpoint_callback,
               best_checkpoint_callback]
    )

# print("Trainer Ready, Tuning Starts!")
# tuner = trainer.tune(model)
# print("Tuner Results: ", tuner)

# new_batch_size = tuner.scale_batch_size(model,)
# model.hparams.batch_size = new_batch_size
# model.hparams.lr = tuner
# lr_finder = trainer.tuner.lr_find(model)

# print("Training Starts!")
# trainer.fit(model)

model.load_from_checkpoint("./CodeT5/ModelCheckpoint/t5model-rewritten-latest-epoch=199-training_loss=0.85.ckpt")

model.eval()
itr = iter(test_dataloader)
batch = next(itr)
input_ids = batch['input_ids']

outputs = model.model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
output = tokenizer.decode(outputs[0], skip_special_tokens=True,
                          clean_up_tokenization_spaces=True)
print(output)