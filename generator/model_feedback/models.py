import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, ElectraTokenizer, ElectraModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import json
import ast

class Tokenizer:
    def __init__(self, args):
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_generator)
        self.max_len_context = args.max_len_context
        self.max_len_answer = args.max_len_answer
        self.max_len_question = args.max_len_question
        self.max_len_persona = args.max_len_persona
        self.task = args.task
        
    def __call__(self, knowledge=None, question=None, persona = None, answer=None,):
        if answer:
            if type(answer) is not list:
                answer = [answer]
            batch_answer = self.tokenizer(answer, max_length=self.max_len_answer, padding='max_length', truncation=True, return_tensors='pt')
            return batch_answer.input_ids, batch_answer.attention_mask

        else:
            if type(knowledge) is not list:
                knowledge = [knowledge]
            if type(question) is not list:
                question = [question]
            if type(persona) is not list:
                persona = [persona]
            
                
            # add separate token to answers
            knowledge = [self.task + ' ' + c for c in knowledge]
            persona = [' ' + str(per) for per in persona]
            question = [' ' + str(ques) for ques in question]
            
            batch_knowledge = self.tokenizer(knowledge, max_length=self.max_len_context, padding='max_length', truncation=True)
            batch_question = self.tokenizer(question, max_length=self.max_len_persona, padding='max_length', truncation=True)
            batch_persona = self.tokenizer(persona, max_length=self.max_len_question, padding='max_length', truncation=True)
            
            input_ids = torch.LongTensor([k[:-1] + p + q for (k, p, q) in zip(batch_knowledge.input_ids, batch_persona.input_ids, batch_question.input_ids)])
            attention_mask = torch.FloatTensor([k[:-1] + p + q for (k, p, q) in zip(batch_knowledge.attention_mask, batch_persona.attention_mask, batch_question.attention_mask)])
            # print(input_ids.shape)
            return input_ids, attention_mask
    
       
class SquadDataset(Dataset):
    def __init__(self, args, train=True):
        self.tokenizer = Tokenizer(args)
        if train:
            self.dataset = pd.read_csv(args.train_dataset)
        else:
            self.dataset = pd.read_csv(args.val_dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        raw = self.dataset.iloc[idx]
        knowledge, question, answer, persona = raw['ground_knowledge'], raw['question_rewritten'], raw['answer'], raw['ground_persona']
        persona = " ".join(ast.literal_eval(persona))
        if persona is None:
          persona = " "
        return self.tokenizer(knowledge=knowledge, question=question, persona=persona), self.tokenizer(answer=answer)


class SquadDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.args = args 
        self.tokenizer = Tokenizer(args)
    
    def prepare_data(self):
        self.train_dataset = SquadDataset(self.args, train=True)
        self.test_dataset = SquadDataset(self.args, train=False)
        
    def setup(self, stage):
        val_length = int(self.args.val_test_split_factor * len(self.test_dataset))
        self.val_dataset, self.test_dataset = random_split(self.test_dataset, [val_length, len(self.test_dataset) - val_length])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.args.num_workers, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.args.num_workers, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.args.num_workers, batch_size=self.batch_size)
    

class LitQGModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.automatic_optimization = False
        self.args = args
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = Evaluator(args, self.device_)
        self.generator = T5ForConditionalGeneration.from_pretrained(args.model_generator)
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_generator)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.gamma = args.gamma
        self.lr = args.learning_rate
        self.automatic_optimization = False 
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler' : optim.lr_scheduler.StepLR(optimizer, self.args.lr_scheduler_step, gamma=self.args.lr_scheduler_factor),
            'interval' : 'step',
            'frequency' : 1,
            'strict' : True
        }
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        # forward
        loss = self.compute_loss(batch, batch_idx)
        
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        
        # backward
        self.manual_backward(loss)
                
        if (batch_idx + 1) % self.args.accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad() # It is good practice to call optimizer.zero_grad() before self.manual_backward(loss).
            
            # # lr scheduler        
            # scheduler.step(loss)

            self.log('train_loss', loss, prog_bar=True)
            return loss
        

    def forward(self, input_ids, attention_mask):
        output = self.generator.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5)
        return output
    
    def compute_loss(self, batch, batch_idx):
        input_ids, attention_mask = batch[0][0].squeeze(1), batch[0][1].squeeze(1)
        decoder_input_ids, decoder_attention_mask = batch[1][0].squeeze(1), batch[1][1].squeeze(1)
        out = self.generator(input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask, labels=decoder_input_ids)
        loss, logits = out.loss, out.logits
        prob = F.softmax(logits, dim=-1)
        batch_size = input_ids.shape[0]
        loss_tune = torch.zeros(batch_size)
        
        for i in range(batch_size):
            prediction = self.tokenizer.decode(prob[i, :, :].argmax(dim=-1).tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            ground_truth = self.tokenizer.decode(decoder_input_ids[i, :].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            persona = self.tokenizer.decode(input_ids[i, self.args.max_len_context-2:self.args.max_len_context+64].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # print(persona)
            r = 1 - self.evaluator(prediction, ground_truth, persona)
            loss_tune[i] = r * self.cross_entropy_loss(logits[i], decoder_input_ids[i])
        
        loss_tune = loss_tune.mean()
        # print("Loss:", loss, "LossTune:", loss_tune)
        loss = loss * self.gamma + (1 - self.gamma) * loss_tune
        # print("Final Loss:", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log('val_loss', loss, on_step=True)
        return loss
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('ptl/val_loss', avg_loss)


class Evaluator:
    def __init__(self, args, device):
        self.args = args
        self.tokenizer = ElectraTokenizer.from_pretrained(args.model_evaluator)
        self.model = ElectraModel.from_pretrained(args.model_evaluator).eval().to(device)
        self.alpha = 0.5
        self.beta = 0.25
        self.delta = 0.25
        
        self.device = device
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def scale_reward(self, reward):
        return (reward + 1 - (self.alpha+self.beta + self.delta)) / (2 - (self.alpha+self.beta + self.delta))
        
    def __call__(self, sen1, sen2, persona):
        with torch.no_grad():
            input_ids = self.tokenizer([sen1.lower(), sen2.lower(), persona.lower()], return_tensors='pt', truncation=True, padding=True).to(self.device)
            out = self.model(**input_ids)['last_hidden_state']
            sim = self.cos(out[0, 0, :], out[1, 0, :]).item()
            persona_sim = self.cos(out[0, 0, :], out[2, 0, :]).item()
            sen1_tokens = word_tokenize(sen1)
            sen2_tokens = word_tokenize(sen2)
            bleu = sentence_bleu([sen2_tokens], sen1_tokens)
            print(f'BLEU: {bleu}, SIM: {sim}, PERSONA_SIM: {persona_sim}')
            reward = self.alpha * bleu + self.beta * sim + self.delta * persona_sim
            reward = self.scale_reward(reward)
            print(f'Reward: {reward}')
            return reward