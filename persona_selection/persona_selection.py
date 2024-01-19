# data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import time
from pprint import pprint
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Union
from tqdm import tqdm

import torch
import numpy as np # linear algebra
import pandas as pd 
from datasets import load_dataset, load_metric, Dataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer, IntervalStrategy, EarlyStoppingCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from torch import nn
import evaluate

model_checkpoint = "xlnet-base-cased"
batch_size = 4
dataset = load_dataset("kanak8278/focus_persona_selection")
counter = dict(Counter(dataset['train']['label']))
label_names= counter.keys()

metric = evaluate.load("seqeval")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint, num_labels=6)

persona_candidate_columns = ["persona1", "persona2", "persona3", "persona4", "persona5", "persona6"] #persona6 for none of these 
# persona_grounding_columns = ["persona_grounding1", "persona_grounding2", "persona_grounding3", "persona_grounding4", "persona_grounding5"] #persona_grounding6 for none of these 

def preprocess_function(examples):

  first_sentences = [[f"{query} {hit_knowledge}"]*6 for query, hit_knowledge in zip(examples['query'], examples['hit_knowledge'])]
  second_sentences = [[examples[persona_candidate_column][i] for persona_candidate_column in persona_candidate_columns]for i, _ in enumerate(examples['dialogID'])]
  first_sentences = sum(first_sentences, [])
  second_sentences = sum(second_sentences, [])

  tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
  return {k: [v[i:i+6] for i in range(0, len(v), 6)] for k, v in tokenized_examples.items()}

dataset_encoded = dataset.map(preprocess_function, batched=True)

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}
# def compute_metrics(eval_preds):
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)

#     # Remove ignored index (special tokens) and convert to labels
#     true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
#     true_predictions = [
#         [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
#     return {
#         "precision": all_metrics["overall_precision"],
#         "recall": all_metrics["overall_recall"],
#         "f1": all_metrics["overall_f1"],
#         "accuracy": all_metrics["overall_accuracy"],
#     }


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        labels = list(map(int, labels))
        # print(labels)
        batch["labels"] = torch.tensor(labels)
        # batch["labels"] = torch.tensor(labels, dtype=torch.float64)
        return batch

# weights = []
weights =[0.848, 0.891, 0.937, 1.078, 1.216, 0.308]
early_stopping_callback = EarlyStoppingCallback(4, 0.001)
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        
        logits = outputs.get("logits")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights)).to(device)
        # print("We are here", self.model.config.num_labels)
        # print(logits.view(-1, self.model.config.num_labels).shape, labels.view(-1).shape)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
if __name__ == "__main__":
    

    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        f"{model_name}_focus_single_weighted_",
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=250, # Evaluate every half epoch
        # evaluation_strategy = IntervalStrategy(),
        learning_rate=5e-7,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        weight_decay=0.01,
        gradient_accumulation_steps=4,
        save_total_limit = 1,
        load_best_model_at_end=True,
        push_to_hub=True,
    )
    trainer = CustomTrainer(
        model,
        args,
        train_dataset=dataset_encoded['train'],
        eval_dataset=dataset_encoded['validation'],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
        callbacks = [early_stopping_callback]
    )
    
    trainer.train()
    
    print(trainer.evaluate(dataset_encoded['test']))
    
    trainer.push_to_hub(commit_message=f"finetuned with weighted cross entropy")
    
    
    