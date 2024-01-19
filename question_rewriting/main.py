import torch
import pandas as pd
import ast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.functional import bleu_score
from torchmetrics.functional.text.rouge import rouge_score

from utils import *

def main(train_loc, save_loc):
    tokenizer, model = initialize_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_df = load_data(train_loc)
    
    result_df = generate(train_df, model, tokenizer, device)
    save_results(result_df, save_loc)
    
    # Calculate and print scores
    bert_scores = calculate_bert_score(list(result_df['query']), list(result_df['question_rewritten']))
    bleu_scores = calculate_bleu_score(list(result_df['query']), list(result_df['question_rewritten']))
    rouge_scores = calculate_rouge_score(list(result_df['query']), list(result_df['question_rewritten']))

    print("Avg BERT Score: ", bert_scores)
    print("BLEU Score: ", bleu_scores)
    print("Avg ROUGE Score: ", rouge_scores)

if __name__ == "__main__":
    train_loc = "../data/focus_train_data.csv"
    # val_loc = "../data/focus_val_data.csv"
    # test_loc = "../data/focus_test_data.csv"

    save_loc = "./train_question_rewritten_1.csv"
    main(train_loc, save_loc)
