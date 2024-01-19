import torch
from torch.utils.data import Dataset
import ast
import time
from itertools import chain
from transformers import T5Tokenizer, T5ForConditionalGeneration

SPECIAL_TOKENS = ["<machine>", "<human>", "<persona>", "<knowledge>"]
ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ['<machine>', '<human>', '<persona>', '<knowledge>']}
SPECIAL_TOKENS_MAP = {
    "machine": "<machine>",
    "human": "<human>",
    "persona": "<persona>",
    "knowledge": "<knowledge>"
}


class DialogDataset(Dataset):
    def __init__(self, tokeinzer, df, max_source_length = 1024, max_target_length = 250, dialog_history = False):
        self.df = df
        self.tokenizer = tokeinzer
        self.dialog_history = dialog_history
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        row = self.df.iloc[item]
        query = str(row["query"])
    
        persona = [SPECIAL_TOKENS_MAP["persona"]] + ast.literal_eval(row['ground_persona'])
        knowledge = [SPECIAL_TOKENS_MAP["knowledge"]] + [str(row["ground_knowledge"])]
        answer = row['answer'] 
        
        if self.dialog_history:    
            history = ast.literal_eval(row['dialog_history'])
            history = [SPECIAL_TOKENS_MAP["human"] + s if i % 2 == 0 else
                    SPECIAL_TOKENS_MAP["machine"] + s
                    for i, s in enumerate(history)] + [SPECIAL_TOKENS_MAP["human"] + query]
        
        else:
            history =  [SPECIAL_TOKENS_MAP["human"] + query]
        
        text = persona + knowledge + history
        text = " ".join(text)
        
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
        

        labels = answer_encoding["input_ids"]
        target_ids = labels.squeeze()
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "target_ids": target_ids,
        }
        

def train(model, dataloaders, optimizer, scheduler, epochs, model_dir, save_best=True):
    train_dataloader, val_dataloader, test_dataloader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    best_loss = float("inf")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    
    s_time = time.time()
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1} started")
        total_loss = 0
        model.train()
        
        for i, data in enumerate(train_dataloader):
            
            optimizer.zero_grad()

            output = forward_pass(data, model, device)
            loss = output['loss']
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            total_loss += loss.item()
            
            loss.backward()    
            optimizer.step()
            
            if i%200 == 0:
                print(f"i: {i}")
            # Validate after every 200 mini batches
            if i % 200 == 0:
                print(f"Epoch: {epoch+1}, Batch: {i+1},  Time Elapsed: {(time.time()-s_time)/60 :.2f}")
                print(f'Current Loss: {loss.item()}, AvgLoss: {total_loss/(i+1) :.4f}')
                val_loss = 0
                with torch.no_grad():
                    model.eval()
                    for idx, data in enumerate(val_dataloader):
                        output = forward_pass(data, model, device)
                        loss = output['loss']
                        if torch.cuda.device_count() > 1:
                            loss = loss.mean()
                        val_loss += loss.item()
                        if idx >= 20:
                            break

                avg_val_loss = val_loss / (idx+1)
                print(f"Validation Loss: {avg_val_loss :.4f}, Time Elapsed:{(time.time()-s_time)/60 :.2f}")
                scheduler.step(avg_val_loss)
                print(f"Learning Rate: get_lr {optimizer.param_groups[0]['lr']}")
                print("Best Loss:", best_loss, "Avg Val Loss:", avg_val_loss)
                if avg_val_loss < best_loss and save_best:
                    print(f"Avg Loss improved:{avg_val_loss} from {best_loss}")
                    best_loss = avg_val_loss
                    model.module.save_pretrained(model_dir)
                    print(f"Model Saved: {model_dir}")
                    print()
            if i >= 800:
                break
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch: {epoch+1}/{epochs} completed!")
        print("Epoch: {}/{} , Loss: {:.4f}, Time Elapsed:{:.2f}".format(epoch+1, epochs, avg_loss, (time.time()-s_time)/60))
        
        val_loss = 0
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(val_dataloader):            
                output = forward_pass(data, model, device)
                loss = output['loss']
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                val_loss += loss.item()
                

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss :.4f}, Time Elapsed:{(time.time()-s_time)/60 :.2f}")
        scheduler.step(avg_val_loss)
        print(f"Learning Rate: get_lr {optimizer.param_groups[0]['lr']}")
        

        if avg_val_loss < best_loss and save_best:
            best_loss = avg_val_loss
            model.module.save_pretrained(model_dir)
            print(f"Model Saved: {model_dir}")
            print("********************************************************\n")
        if epoch % 5 == 0:
            inference_model = T5ForConditionalGeneration.from_pretrained(config.MODEL_SAVE_DIR)
            print("Inference Model Ready")
            predictions, actuals = validate(tokenizer, inference_model, test_dataloader)
            print("Predictions Ready!")
            final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
            final_df.to_csv(f"{model_dir}/t5_predictions.csv", index=False)
            print('Output Files generated for review')


def forward_pass(data, model, device, pad_token_id=0,):
    input_ids = data["input_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)
    target_ids = data["target_ids"]
    target_ids = target_ids.to(device)
    y = target_ids.to(device)
    y_ids = y[:, :-1].contiguous()
    lm_labels = y[:, 1:].clone().detach()
    lm_labels[y[:, 1:] == pad_token_id] = -100
    
    output = model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids=y_ids, labels=lm_labels)
    loss = output[0]
    return {
        "loss": loss,
        "output": output
    }


def validate(tokenizer, model, loader, max_length=256):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    s_time = time.time()
    model.eval()
    model.to(device)
    
    predictions = []
    actuals = []
    with torch.no_grad():
          for i, data in enumerate(loader):
            y = data["target_ids"].to(device, dtype = torch.long)
            ids = data["input_ids"].to(device, dtype = torch.long)
            mask = data["attention_mask"].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=max_length, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if i+1%100==0:
                print(f'Completed {i+1}, Time Elapsed: {(time.time() - s_time)/60 :.2f}')
            predictions.extend(preds)
            actuals.extend(target)
    print(f"Validation Complete, Time Elapsed: {(time.time() - s_time)/60 :.2f}")
    return predictions, actuals
    