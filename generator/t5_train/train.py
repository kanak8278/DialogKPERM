import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import wandb
import ast
import os

from utils import DialogDataset, train, validate

# wandb.init(project="t5_training_ground_knowledge")

args = {
    # Initialize config
    "TRAIN_BATCH_SIZE" : 2,    # input batch size for training (default: 64)
    "VALID_BATCH_SIZE" : 16,    # input batch size for testing (default: 1000)
    "TRAIN_EPOCHS" : 2,        # number of epochs to train (default: 10)
    "VAL_EPOCHS" : 1, 
    "LEARNING_RATE" : 5e-5,    # learning rate (default: 0.01)
    "SEED" : 42,               # random seed (default: 42)
    "MAX_LEN" : 1024,
    "ANSWER_LEN" : 256, 
    "MODEL_SAVE_DIR" : "t5_weights/"

}

# wandb.config.update(args)


# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(args["SEED"]) # pytorch random seed
np.random.seed(args["SEED"]) # numpy random seed
torch.backends.cudnn.deterministic = True


SPECIAL_TOKENS = ["<machine>", "<human>", "<persona>", "<knowledge>"]
ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ['<machine>', '<human>', '<persona>', '<knowledge>']}
SPECIAL_TOKENS_MAP = {
    "machine": "<machine>",
    "human": "<human>",
    "persona": "<persona>",
    "knowledge": "<knowledge>"
}


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer)
    # if type(tokenizer).__name__ == 'GPT2Tokenizer':
    #     ATTR_TO_SPECIAL_TOKEN['pad_token'] = '<pad>'
    #     print('<pad> token added!')
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    num_added_tokens = len(SPECIAL_TOKENS)
    print("orig num", orig_num_tokens, "num_added", num_added_tokens) #50265, 4
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def initialize_model_and_tokenizer(model_name="t5-base"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    add_special_tokens_(model, tokenizer)
    return model, tokenizer
def load_data(train_loc, val_loc, test_loc):
    train_df = pd.read_csv(train_loc)
    val_df = pd.read_csv(val_loc)
    test_df = pd.read_csv(test_loc)
    return train_df, val_df, test_df

def setup_training_environment(model):
    if os.path.exists(args['MODEL_SAVE_DIR']):
        args['MODEL_SAVE_DIR'] = "new_weights_dir/"

    num_gpus = torch.cuda.device_count()
    model = torch.nn.parallel.DataParallel(model, device_ids=list(range(num_gpus)), dim=0)
    return model, num_gpus

def create_dataloaders(tokenizer, train_df, val_df, test_df, num_gpus):
    train_params = {
        'batch_size': args["TRAIN_BATCH_SIZE"]*num_gpus,
        'shuffle': False,
        'num_workers': 4
        }

    val_params = {
        'batch_size': args["VALID_BATCH_SIZE"]*num_gpus,
        'shuffle': False,
        'num_workers': 4
        }
    
    # Create the dataloaders
    train_dataloader = DataLoader(DialogDataset(tokenizer, train_df, args["MAX_LEN"], args["ANSWER_LEN"]), **train_params)
    val_dataloader = DataLoader(DialogDataset(tokenizer, val_df, args["MAX_LEN"], args["ANSWER_LEN"]), **val_params)
    test_dataloader = DataLoader(DialogDataset(tokenizer, test_df, args["MAX_LEN"], args["ANSWER_LEN"]), **val_params)
    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader
    }
    print("Train Mini-Batch: ", len(train_dataloader), "Val Mini-Batch: ", len(val_dataloader), "Test Mini-Batch: ", len(test_dataloader))
    
    return dataloaders

def inference(tokenizer, model, test_dataloader):
    predictions, actuals = validate(tokenizer, model, test_dataloader)
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
    final_df.to_csv(f'{args["MODEL_SAVE_DIR"]}/t5_predictions.csv', index=False)
    print('Output Files generated for review')

if __name__ == "__main__":
    model, tokenizer = initialize_model_and_tokenizer()
    train_df, val_df, test_df = load_data("../../data/focus_train_data.csv", "../../data/focus_val_data.csv", "../../data/focus_test_data.csv")
    
    model, num_gpus = setup_training_environment(model)
    dataloaders = create_dataloaders(tokenizer, train_df, val_df, test_df, num_gpus)

    
    # # Initialize the model for training
    # Define the optimizer and scheduler
    # optimizer = torch.optim.AdamW(params =  model.parameters(), lr=args["LEARNING_RATE"])
    # # scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.01, last_epoch=-1)
    
    # # wandb.watch(model, log="all")
    # print("Starting Training!")
    # train(model, dataloaders, optimizer, scheduler, args["TRAIN_EPOCHS"], args["MODEL_SAVE_DIR"], save_best=True)
    # print() origin/remotes/origin/t5_lightning

    inference_model = T5ForConditionalGeneration.from_pretrained("/home/ubuntu/chat_persona/generator/t5_train/t5_weights")
    inference(tokenizer, inference_model, dataloaders["test"])
    
  