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

# Define the dataset and dataloader for training
class DialogDataset(Dataset):
    def __init__(self, tokeinzer, df, max_source_length = 512, max_target_length = 250, dialog_history = False):
        self.df = df
        self.tokenizer = tokenizer
        self.dialog_history = dialog_history
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        row = self.df.iloc[item]
        query = str(row["query"])
        persona = str(" ".join(ast.literal_eval(row['ground_persona'])))
        context = str(row['ground_knowledge'])
        answer = f"{row['answer']} </s>"
        
        if self.dialog_history:
            dialog_history = str(row["dialog_history"])
            text = f"answer_me: {query} history: {dialog_history} context: {context} persona: {persona} </s>"
        else:
            text = f"answer_me: {query} context: {context} persona: {persona} </s>"


        encoding = self.tokenizer.encode_plus(text, 
                                              max_length=self.max_source_length,
                                              padding = 'max_length',
                                              add_special_tokens=True,
                                              truncation=True,
                                              return_tensors="pt")
        answer_encoding = self.tokenizer.encode_plus(answer, 
                                                     max_length=self.max_target_length,
                                                     padding = 'max_length',
                                                     add_special_tokens=True,
                                                     truncation=True,
                                                     return_tensors="pt")
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = answer_encoding["input_ids"]
        target_ids = labels.squeeze()
        return input_ids, attention_mask, target_ids


# Define the training loop
def train(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, model_dir, save_best=True):
    best_loss = float("inf")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    s_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for i, (input_ids, attention_mask, target_ids) in enumerate(train_dataloader):
          optimizer.zero_grad()

          input_ids = input_ids.to(device)
          attention_mask = attention_mask.to(device) 
          target_ids = target_ids.to(device)
          y = target_ids.to(device)
          y_ids = y[:, :-1].contiguous()
          lm_labels = y[:, 1:].clone().detach()
          lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
           
          output = model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids=y_ids, labels=lm_labels)
          loss = output[0]
          
          total_loss += loss.item()
          loss.backward()
          optimizer.step()
          scheduler.step(loss.item())
          
          if i % 100 == 0:
            print(f"Epoch: {epoch+1}, Batch: {i}, Time Elapsed: {time.time()-s_time :.2f}")
                    
          
        avg_loss = total_loss / len(train_dataloader)
        print("Epoch: {}/{}, Loss: {:.4f}, Time Elapsed:{:.2f}".format(epoch+1, epochs, avg_loss, time.time()-s_time))
        
        val_loss = 0
        with torch.no_grad():
          model.eval()
          for i, (input_ids, attention_mask, target_ids) in enumerate(val_dataloader):
            
              input_ids = input_ids.to(device)
              attention_mask = attention_mask.to(device) 
              target_ids = target_ids.to(device)
              y = target_ids.to(device)
              y_ids = y[:, :-1].contiguous()
              lm_labels = y[:, 1:].clone().detach()
              lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
              
              output = model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids=y_ids, labels=lm_labels)
              loss = output[0]
              val_loss += loss.item()


        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss :.4f}, Time Elapsed:{time.time()-s_time :.2f}")
        
        if avg_val_loss < best_loss and save_best:
          best_loss = avg_val_loss
          model.save_pretrained(model_dir)
        #   torch.save(model.state_dict(), "t5_best_model.pt")
        



def validate(tokenizer, model, loader):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    
    wandb.init(project="t5_training_ground_knowledge")

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training  
    config = wandb.config          # Initialize config
    config.TRAIN_BATCH_SIZE = 4    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 8    # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = 4        # number of epochs to train (default: 10)
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
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    print(f"Model Vocab Size: {model.config.vocab_size}, Tokenizer Vocab Size: {tokenizer.vocab_size}")

    # model.config.vocab_size = tokenizer.vocab_size
    # model.resize_token_embeddings(len(tokenizer))

    # Load the training data and split it into tt5_ground_knw_ground_personaraining and validation sets
    train_loc = "./data/focus_train_data.csv" #location to data file
    val = "./data/focus_val_data.csv" #location to data file
    test = "./data/focus_test_data.csv"
    
    train_df = pd.read_csv(train_loc)
    val_df = pd.read_csv(val)
    
    print("TRAIN Dataset: {}".format(train_df.shape))
    print("VAL Dataset: {}".format(val_df.shape))
    
    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 4
        }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 4
        }
    
    # Create the dataloaders
    train_dataloader = DataLoader(DialogDataset(tokenizer, train_df, config.MAX_LEN, config.ANSWER_LEN), **train_params)
    val_dataloader = DataLoader(DialogDataset(tokenizer, val_df, config.MAX_LEN, config.ANSWER_LEN), **val_params)
    
    print(len(train_dataloader), len(val_dataloader))
    # Define the optimizer and scheduler
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    # Log metrics with wandb
    wandb.watch(model, log="all")
    
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, config.TRAIN_EPOCHS, config.MODEL_SAVE_DIR)
    
    inference_model = T5ForConditionalGeneration.from_pretrained(config.MODEL_SAVE_DIR)
    test_df = pd.read_csv(test)
    print("TEST Dataset: {}".format(test_df.shape))

    test_dataloader = DataLoader(DialogDataset(tokenizer, test_df, config.MAX_LEN, config.ANSWER_LEN), **val_params)
    print(len(test_dataloader))
    predictions, actuals = validate(tokenizer, inference_model, test_dataloader)
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
    final_df.to_csv(f'{config.MODEL_SAVE_DIR}/predictions.csv', index=False)
    print('Output Files generated for review')