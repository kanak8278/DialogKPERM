import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import ast
import gc
import time
import wandb


def validate(tokenizer, model, device, loader):
    s_time = time.time()
    model.eval()
    model.to(device)
    
    predictions = []
    actuals = []
    with torch.no_grad():
          for i, (input_ids, attention_mask, target_ids) in enumerate(loader):
            y = target_ids.to(device, dtype = torch.long)
            ids = input_ids.to(device, dtype = torch.long)
            mask = attention_mask.to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=250, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if i%30==0:
                print(f'Completed {i}, Time Elapsed: {time.time() - s_time :.2f}')
            predictions.extend(preds)
            actuals.extend(target)
    print(f"Validation Complete, Time Elapsed: {time.time() - s_time :.2f}")
    return predictions, actuals

if __name__ == "__main__":
    config.TRAIN_BATCH_SIZE = 2    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 2    # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = 2        # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1 
    config.LEARNING_RATE = 5e-5    # learning rate (default: 0.01)
    config.SEED = 42               # random seed (default: 42)
    config.MAX_LEN = 512
    config.ANSWER_LEN = 250 
    config.MODEL_SAVE_DIR = "t5_grd_knw_grd_persona/"
    
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED) # pytorch random seed
    np.random.seed(config.SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

    
    
    # Load the T5 tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    inference_model = T5ForConditionalGeneration.from_pretrained(config.MODEL_SAVE_DIR)
    
    test = "/work/kanakr/chat_persona/data/focus_val_data.csv"
    test_df = pd.read_csv(test)
    print("TEST Dataset: {}".format(test_df.shape))

    test_dataloader = DataLoader(DialogDataset(test_df, config.MAX_LEN, config.ANSWER_LEN), **val_params)
    predictions, actuals = validate(tokenizer, inference_model, device, test_dataloader)
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
    final_df.to_csv(f'{config.MODEL_SAVE_DIR}/predictions.csv')
    print('Output Files generated for review')