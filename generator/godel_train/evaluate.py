import pandas as pd
import transformers
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    set_seed,
)

from utils import GodelDialogDataset, validate

args = {
    "USE_WANDB": False,
    "wandb_exp_name": "Personalized Response Generation",
    "exp_name": "godel_training",
    "config_name" : None,
    "model_name_or_path": "microsoft/GODEL-v1_1-base-seq2seq",
    "tokenizer_name": "microsoft/GODEL-v1_1-base-seq2seq",
    "use_slow_tokenizer": False,
    "pad_to_max_length":2048 ,
    "max_target_length":512 ,
    "weight_decay":1e-4,
    "TRAIN_BATCH_SIZE" : 2,    # input batch size for training (default: 64)
    "VALID_BATCH_SIZE" : 16,    # input batch size for testing (default: 1000)
    "TRAIN_EPOCHS" : 20,        # number of epochs to train (default: 10)
    "VAL_EPOCHS" : 1, 
    "LEARNING_RATE" : 5e-5,    # learning rate (default: 0.01)
    "SEED" : 42,               # random seed (default: 42)
    "MODEL_SAVE_DIR": "./weights/"
}

#  Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(args["SEED"]) # pytorch random seed
np.random.seed(args["SEED"]) # numpy random seed
torch.backends.cudnn.deterministic = True

val_params = {
        'batch_size': args["VALID_BATCH_SIZE"]*1,
        'shuffle': True,
        'num_workers': 4
        }
if __name__ == "__main__":
    test = "../../data/focus_test_data.csv"
    test_df = pd.read_csv(test)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(args["MODEL_SAVE_DIR"])
    tokenizer = AutoTokenizer.from_pretrained(args["model_name_or_path"], use_fast=not args["use_slow_tokenizer"])
    
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    test_dataloader = DataLoader(GodelDialogDataset(tokenizer, test_df, args["pad_to_max_length"], args["max_target_length"]), **val_params)
    
    print("Test Mini-Batch: ", len(test_dataloader))
    print("Evaluation starts on Test Data:")
    predictions, actuals = validate(tokenizer, model, test_dataloader)
    print("Predictions Ready!")
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
    final_df.to_csv(f'godel_predictions_01.csv', index=False)
    print('Output Files generated for review')

