import torch
import pandas as pd
import ast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.functional import bleu_score
from torchmetrics.functional.text.rouge import rouge_score

def initialize_model(model_name="castorini/t5-base-canard"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def load_data(file_location):
    return pd.read_csv(file_location)

def save_results(df, file_location):
    df.to_csv(file_location, index=False)
    print("File Saved!")

def generate_utils(model, tokenizer, history, query, device):
    history = str(" ||| ".join(ast.literal_eval(history)))
    text = history + " ||| " + query
    input_ids = tokenizer(text, return_tensors='pt').to(device)
    generated_ids = model.generate(input_ids=input_ids['input_ids'], attention_mask=input_ids['attention_mask'])
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids][0]
    return preds, text

def generate(df, model, tokenizer, device):
    question_rewritten, texts = [], []
    for index, row in df.iterrows():
        if index % 100 == 0:
            print(index)
        new_question, text = generate_utils(model, tokenizer, row['dialog_history'], row['query'], device)
        question_rewritten.append(new_question)
        texts.append(text)
    df['question_rewritten'] = question_rewritten
    return df

def calculate_bert_score(queries, rewritten_questions):
    scores = bert_score(rewritten_questions, queries)
    avg_scores = {metric: sum(scores[metric]) / len(scores[metric]) for metric in scores.keys()}
    return avg_scores

def calculate_bleu_score(queries, rewritten_questions):
    return bleu_score(rewritten_questions, queries)

def calculate_rouge_score(queries, rewritten_questions):
    scores = rouge_score(rewritten_questions, queries)
    avg_scores = {metric: sum(scores[metric]) / len(scores[metric]) for metric in scores.keys()}
    return avg_scores