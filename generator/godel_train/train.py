import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import transformers
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

from utils import GodelDialogDataset, train, validate


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
    "VALID_BATCH_SIZE" : 8,    # input batch size for testing (default: 1000)
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

if args["USE_WANDB"]:
        config = dict(
        dataset_id = "DialoGLM",
        infra = "aml",
        )
        import wandb
        
        wandb.init(
        project=args["wandb_exp_name"],
        notes="Finetuning",
        tags=["DialoGLM", "GODEL"],
        config=config,
        entity= 'DialoGLM')
        wandb.run.name = args["exp_name"]
        wandb.config.update(args)
        





if __name__ == "__main__":
        
    if args["config_name"]:
            config = AutoConfig.from_pretrained(args["config_name"])
    elif args["model_name_or_path"]:
        config = AutoConfig.from_pretrained(args["model_name_or_path"])

    if args["tokenizer_name"]:
            tokenizer = AutoTokenizer.from_pretrained(args["tokenizer_name"], use_fast=not args["use_slow_tokenizer"])
    elif args["model_name_or_path"]:
        tokenizer = AutoTokenizer.from_pretrained(args["model_name_or_path"], use_fast=not args["use_slow_tokenizer"])

    if args["model_name_or_path"]:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args["model_name_or_path"],
                config=config,
            )


    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    padding = "max_length" if args["pad_to_max_length"] else False
    max_target_length = args["max_target_length"]

    num_gpus = torch.cuda.device_count()
    print("Total GPU Count:", num_gpus)
    model = torch.nn.parallel.DataParallel(model, device_ids=list(range(num_gpus)), dim=0)
    
    train_params = {
        'batch_size': args["TRAIN_BATCH_SIZE"]*num_gpus,
        'shuffle': False,
        'num_workers': 4
        }

    val_params = {
        'batch_size': args["VALID_BATCH_SIZE"]*num_gpus,
        'shuffle': True,
        'num_workers': 4
        }
    
    
    train_loc = "../../data/focus_train_data.csv" #location to data file
    val = "../../data/focus_val_data.csv" #location to data file
    test = "../../data/focus_test_data.csv"
    
    train_df = pd.read_csv(train_loc)
    val_df = pd.read_csv(val)
    test_df = pd.read_csv(test)
    
    print("TRAIN Dataset: {}".format(train_df.shape))
    print("VAL Dataset: {}".format(val_df.shape))
    print("TEST Dataset: {}".format(test_df.shape))
    
    
    # Create the dataloaders
    train_dataloader = DataLoader(GodelDialogDataset(tokenizer, train_df, args["pad_to_max_length"], args["max_target_length"]), **train_params)
    val_dataloader = DataLoader(GodelDialogDataset(tokenizer, val_df, args["pad_to_max_length"], args["max_target_length"]), **val_params)
    test_dataloader = DataLoader(GodelDialogDataset(tokenizer, test_df, args["pad_to_max_length"], args["max_target_length"]), **val_params)
    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader
    }
    
    print("Train Mini-Batch: ", len(train_dataloader), "Val Mini-Batch: ", len(val_dataloader), "Test Mini-Batch: ", len(test_dataloader))
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=0.01, last_epoch=-1)
    
    print("Starting Training!")
    train(model, dataloaders, optimizer, scheduler, args["TRAIN_EPOCHS"], args["MODEL_SAVE_DIR"], save_best=True)
    print("Done")
    
    # inference_model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_SAVE_DIR)
    # print("Inference Model Ready")
    # predictions, actuals = validate(tokenizer, inference_model, test_dataloader)
    # print("Predictions Ready!")
    # final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
    # final_df.to_csv(f"./godel_predictions.csv", index=False)
    # print('Output Files generated for review')
