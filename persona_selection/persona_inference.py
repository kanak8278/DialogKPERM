# data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import time
from pprint import pprint
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Union
from tqdm import tqdm

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from torchmetrics.functional import bleu_score
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.text.bert import BERTScore
import torch
import numpy as np # linear algebra
import pandas as pd 
from datasets import load_dataset, load_metric, Dataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer, IntervalStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForMultipleChoice

def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
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

def preprocess_function(examples, return_tensors=None):
  for key in examples.keys():
    if examples[key] is not list():
      examples[key] = [examples[key]]

  first_sentences = [[f"{query} {hit_knowledge}"]*6 for query, hit_knowledge in zip(examples['query'], examples['hit_knowledge'])]
  second_sentences = [[examples[persona_candidate_column][i] for persona_candidate_column in persona_candidate_columns]for i, _ in enumerate(examples['dialogID'])]
  first_sentences = sum(first_sentences, [])
  second_sentences = sum(second_sentences, [])
  
  tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, padding=True, return_tensors=return_tensors)

  return {k: [v[i:i+6] for i in range(0, len(v), 6)] for k, v in tokenized_examples.items()}

# dataset_encoded = dataset.map(preprocess_function, batched=True)



if __name__ == "__main__":
  tokenizer = AutoTokenizer.from_pretrained("/work/kanakr/chat_persona/xlnet-base-cased-focus_single_persona_weighted_entropy")
  model = AutoModelForMultipleChoice.from_pretrained("/work/kanakr/chat_persona/xlnet-base-cased-focus_single_persona_weighted_entropy")
  model = model.cuda()
  model.eval()
  persona_candidate_columns = ["persona1", "persona2", "persona3", "persona4", "persona5", "persona6"] #persona6 for none of these 
  dataset = load_dataset("kanak8278/focus_persona_selection")

  
  examples = dataset['test'][0]
  # if type(examples) is not list:
  #   examples = [examples]
  print(examples.keys())
  
  
  results = []
  for data in tqdm(dataset['test']):
    
    hit_knowledge = data['hit_knowledge']
    persona1 = data['persona1']
    persona2 = data['persona2']
    persona3 = data['persona3']
    persona4 = data['persona4']
    persona5 = data['persona5']
    persona6 = data['persona6']
    label = data['label']

    inputs = preprocess_function(data.copy(), 'pt')
    outputs = model(**{k: v[0].unsqueeze(0).cuda()  for k, v in inputs.items() if k != "token_type_ids"})
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    req_columns = ['dialogID', 'utterance', 'label','query']
    result = {key:data[key] for key in req_columns}
    result['true_persona'] = data[f"persona{data['label']+1}"]
    result['pred_persona'] = data[f"persona{predicted_class+1}"]
    result['pred_class'] = predicted_class
    results.append(result)

  result_df = pd.DataFrame(results)
  
  result_df.to_csv("test_persona_results_2.csv", index=False)
  
  df = pd.read_csv("/work/kanakr/chat_persona/test_persona_results_2.csv")
  print(df.columns)
  preds = list(df['pred_class'])
  ground = list(df['label'])
  
  # {'accuracy': 0.777302174919019, 'f1': 0.777302174919019, 'precision': 0.777302174919019, 'recall': 0.777302174919019}
  # {'accuracy': 0.7341508560851457, 'f1': 0.7341508560851457, 'precision': 0.7341508560851457, 'recall': 0.7341508560851457}
  

  """Metrics Calculation"""
  print(compute_metrics(preds, ground))
  
  # print("Metrics evaluation starting:")
  # print("===================================================================================")
  
  # print("Calculating Bleu Score")
  # bleu = bleu_score(preds[:], ground[:])
  # print("Bleu Score: ", bleu)
  # print("===================================================================================")
  
  # print("Calculating Rouge Score")
  # rouge = rouge_score(preds[:], ground[:])
  # print("Rouge Score: ", rouge)
  # print("===================================================================================")
  
  # bertscore = BERTScore('bert-base-uncased')
  # print("Calculating BERT Score")
  # bert = bertscore(preds[:], ground[:])
  # bert = {k: sum(vv)/len(vv) for k, vv in bert.items()}
  # print("BERT Score: ", bert)
  # print("===================================================================================")