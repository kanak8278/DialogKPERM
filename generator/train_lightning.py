import ast
import torch

class DialogDataset(Dataset):
    def __init__(self, df, tokenizer, max_source_length = 512,
                 max_target_length = 128, dialog_history = False):
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
        context = str(row["ground_knowledge"])
        answer = str(row["answer"])


        if self.dialog_history:
          dialog_history = str(row["dialog_history"])
          text = dialog_history + " " + query + " " +  persona + " " + context
        else:
          text = query +  " " + persona + " " + context

        source_encoding = self.tokenizer(
                                text,
                                max_length=self.max_source_length,
                                padding = 'max_length',
                                truncation=True,
                                return_attention_mask = True,
                                add_special_tokens=True,
                                return_tensors="pt")
        target_encoding = self.tokenizer.encode_plus(
                                answer,
                                max_length=self.max_target_length,
                                padding = 'max_length',
                                return_attention_mask = True,
                                add_special_tokens=True,
                                truncation=True,
                                return_tensors="pt")

        input_ids = source_encoding["input_ids"].flatten()
        attention_mask = source_encoding["attention_mask"].flatten()

        labels = target_encoding["input_ids"]
        labels[labels == 0] = -100
        target_ids = labels.flatten()
        # print(input_ids.shape, attention_mask.shape, labels.shape)
        return {
            "query": query,
            "persona": persona,
            "context": context,
            "answer": answer,
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "target_ids": target_ids}

class DialogDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer,
        batch_size = 4,
        max_source_length = 512,
        max_target_length = 128,
        dialog_history = False
    ):
        super().__init__()
        self.bs = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.dialog_history = dialog_history
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def setup(self,):
        self.train_dataset = DialogDataset(
            self.train_df,
            self.tokenizer,
            self.max_source_length,
            self.max_target_length,
            self.dialog_history
            )
        self.test_dataset = DialogDataset(
            self.test_df,
            self.tokenizer,
            self.max_source_length,
            self.max_target_length,
            self.dialog_history
            )

    def train_dataloader(self,):
        return DataLoader(
            self.train_dataset,
            batch_size = self.bs,
            shuffle = False,
            num_workers = 4
            )
    def val_dataloader(self,):
        return DataLoader(
            self.test_dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = 4
            )
class DialogModel(pl.LightningModule):
    def __init__(self, MODEL_NAME):
        super().__init__()
        self.model = T5forConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

    def forward(self, input_ids, attention_mask, labels=None):
        output = model(
            input_ids= input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
        return output.loss, output.logits
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['target_ids']
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar = True, logger=True)
        return loss
    def val_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['target_ids']
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar = True, logger=True)
        return loss
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr = 0.0001)


checkpoint_callback = ModelCheckpoint(
    dirpath = "checkpoints",
    filename = "best-checkpoint",
    save_top_k = 1,
    verbose= True,
    monitor = "val_loss",
    mode = "min"
)