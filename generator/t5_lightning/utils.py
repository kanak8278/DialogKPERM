import pandas as pd
import ast
from datasets import Dataset, load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from torch.utils.data import DataLoader



SPECIAL_TOKENS = [
    # "<machine>", "<human>",
    "<persona>", "<knowledge>", "<query>"]
ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': [
    # '<machine>', '<human>',
    '<persona>', '<knowledge>', '<query>']}
SPECIAL_TOKENS_MAP = {
    # "machine": "<machine>",
    # "human": "<human>",
    "persona": "<persona>",
    "knowledge": "<knowledge>",
    "query": "<query>"
}

COLUMNS = ['query', 'answer', 'ground_knowledge', 'ground_persona']

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer)
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    num_added_tokens = len(SPECIAL_TOKENS)
    print("orig num", orig_num_tokens, "num_added", num_added_tokens)
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def ast_literal(example):
    return " </s> ".join(ast.literal_eval(example))


def preprocess_examples(examples):
    query = SPECIAL_TOKENS_MAP["query"] +" "+ str(examples["query"])
    persona = SPECIAL_TOKENS_MAP["persona"] + " "+ str(examples['ground_persona'])
    knowledge = SPECIAL_TOKENS_MAP["knowledge"] + " " + str(examples["ground_knowledge"])
    answer = examples['answer']  
    text =  query + " " + persona + " " + knowledge
    
    model_inputs = tokenizer.encode_plus(text, 
                                    max_length= 1024,
                                    padding = 'max_length',
                                    add_special_tokens=True,
                                    truncation=True,
                                    # return_tensors="pt"
                                    )
    labels = tokenizer.encode_plus(answer, 
                                    max_length= 256,
                                    padding = 'max_length',
                                    add_special_tokens=True,
                                    truncation=True,
                                    # return_tensors="pt"
                                    ).input_ids
    # print(model_inputs)
    model_inputs["labels_raw"] = labels.copy()
    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels[labels == tokenizer.pad_token_id] = -100
    # labels_with_ignore_index = []
    # for labels_example in labels:
    #     print(type(labels_example))
    #     labels_example = [label if label != 0 else -100 for label in labels_example]
    #     labels_with_ignore_index.append(labels_example)

    model_inputs["labels"] = labels

    return model_inputs

def demo(df):
    row = df.iloc[5]
    model_inputs = preprocess_examples(row)
    # print(model_inputs['input_ids'])
    input_text = [tokenizer.decode(g, 
                              skip_special_tokens=True, 
                              clean_up_tokenization_spaces=True
                              ) for g in model_inputs['input_ids']]
    output_text = [tokenizer.decode(g, 
                            skip_special_tokens=True, 
                          clean_up_tokenization_spaces=True
                            ) for g in model_inputs['labels_raw']]
    # tokenizer.decode(model_inputs['input_ids'])
    print("Decoded:")
    print("Input: ", input_text)
    print("Output: ", output_text)


class CodeT5(pl.LightningModule):
    def __init__(self, lr=5e-5, num_train_epochs=15, warmup_steps=1000):
        super().__init__()
        # self.model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):     
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss
      
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     

        return loss

    def configure_optimizers(self):
        # create optimizer
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # create learning rate scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * len(train_dataloader)
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return valid_dataloader

    def test_dataloader(self):
        return test_dataloader


if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    
    print(f"Model Vocab Size: {model.config.vocab_size}",
          f"Tokenizer Vocab Size: {tokenizer.vocab_size}")
    add_special_tokens_(model, tokenizer)
    print(f"Model Vocab Size: {model.config.vocab_size}, Tokenizer Vocab Size: {tokenizer.vocab_size}")
    
    
    data_files = {
        "train": "../../data/focus_train_data.csv",
        "validation": "../../data/focus_val_data.csv",
        "test": "../../data/focus_test_data.csv",  
    }
    train_df = pd.read_csv(data_files["train"])[COLUMNS]
    val_df = pd.read_csv(data_files["validation"])[COLUMNS]
    test_df = pd.read_csv(data_files["test"])[COLUMNS]
    
    train_df['ground_persona'] = train_df['ground_persona'].apply(ast_literal)
    demo(train_df)
    # print(train_df['ground_persona'][15])
    
    train_dataset = Dataset.from_pandas(train_df)
    # val_dataset = Dataset.from_pandas(val_df)
    # test_dataset = Dataset.from_pandas(test_df)
    
    # dataset = load_dataset('csv', data_files = {
    #     "train": "../../data/focus_train_data.csv",
    #     "validation": "../../data/focus_val_data.csv",
    #     "test": "../../data/focus_test_data.csv",  
    # })
    # print(train_dataset)
    train_dataset = train_dataset.map(preprocess_examples, batched=True)
    # train_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    # train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1)
    
    # val_dataset = val_dataset.map(preprocess_examples, batched=True)
    # val_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    # val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=8)
    
    # test_dataset = test_dataset.map(preprocess_examples, batched=True)
    # test_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    # test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8)
    
    
    # print(train_dataloader, val_dataloader, test_dataloader)

    
    

    