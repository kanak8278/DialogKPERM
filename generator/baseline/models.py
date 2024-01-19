import json
import pandas as pd
import numpy as np
import torch
from itertools import chain
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu
# from termcolor import colored
import textwrap
import ast
from transformers import (
    BartTokenizer,
    ElectraTokenizer,
    ElectraModel,
    BartForConditionalGeneration,
    BartConfig,
    GPT2Tokenizer,
    GPTNeoForCausalLM  
)

from tqdm.auto import tqdm
import matplotlib.pyplot as plt 
from matplotlib import rcParams, rc

pl.seed_everything(42)


class Tokenizer:
    def __init__(self, args):
        self.args = args
        self.tokenizer = BartTokenizer.from_pretrained(args.model_generator)
        # self.tokenizer = GPT2Tokenizer.from_pretrained(
        #     "EleutherAI/gpt-neo-1.3B",    
        #     bos_token='<|startoftext|>',
        #     eos_token='<|endoftext|>',
        #     pad_token='<|pad|>')
        
        self.max_len_context = args.max_len_context
        self.max_len_answer = args.max_len_answer
        self.max_len_question = args.max_len_question
        self.max_len_persona = args.max_len_persona
        self.max_len_history = args.max_len_history
        self.history_size = args.history_size
        self.task = args.task

        
    def __call__(self, knowledge=None, question=None, persona = None, answer=None, history=None, **kwargs):
        if answer:
            if type(answer) is not list:
                answer = [answer]
            batch_answer = self.tokenizer(answer,
                                          max_length=self.max_len_answer,
                                          padding='max_length',
                                          truncation=True,
                                          add_special_tokens=True,
                                          return_attention_mask=True,
                                          return_tensors='pt')
            labels = batch_answer.input_ids.clone()
            # labels[labels==0] = -100
            return labels, batch_answer.attention_mask

        else:
            batch = {'knowledge': None,
                     'persona': None,
                     'history': None,
                     'question': None}
            
            if type(question) is not list:
                question = [question]

            question = [' ' + str(ques) for ques in question]
            batch['question'] = self.tokenizer(question,
                                            max_length=self.max_len_persona,
                                            padding='max_length',
                                            truncation=True,
                                            add_special_tokens=True)
           
            if self.args.use_persona:
                if type(persona) is not list:
                    persona = [persona]
                persona = [' ' + str(per) for per in persona]
                batch['persona'] = self.tokenizer(persona,
                                           max_length=self.max_len_question,
                                           padding='max_length',
                                           truncation=True,
                                           add_special_tokens=True)
            
            if self.args.use_knowledge:
                if type(knowledge) is not list:
                    knowledge = [knowledge]
                knowledge = [self.task + ' ' + c for c in knowledge] 
                batch['knowledge'] = self.tokenizer(knowledge,
                                             max_length=self.max_len_context,
                                             padding='max_length',
                                             truncation=True,
                                             add_special_tokens=True)
                
                
            if self.args.use_history:
                if type(history) is not list:
                    history = [history]
                history = [' ' + str(his) for his in history]
                batch['history'] = self.tokenizer(history,
                                                 max_length=self.max_len_history,
                                                 padding='max_length',
                                                 truncation=True,
                                                 add_special_tokens=True)
                
            
            input_ids = {key:batch[key].input_ids for key in batch.keys() if batch[key] is not None}
            input_ids = torch.LongTensor([list(chain(*z)) for z in zip(*input_ids.values())])
            
            attention_mask = {key:batch[key].attention_mask for key in batch.keys() if batch[key] is not None}
            attention_mask = torch.FloatTensor([list(chain(*z)) for z in zip(*attention_mask.values())])
            
            return input_ids, attention_mask, persona
    
       
class FocusDataset(Dataset):
    def __init__(self, args, train=True):
        self.args = args
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
        knowledge, history, persona = None, None, None
        question, answer  =  raw['query'], raw['answer']
        
        if self.args.use_knowledge:
            knowledge = raw['hit_knowledge']   
        
        if self.args.use_history:
            history = ast.literal_eval(raw['dialog_history'])
            if type(history) is not list or history is None or history == []:
                history = " "
            else:
                history_size = min (self.args.history_size,  len(history)) 
                history = history[-history_size:]
                history = " ".join(history)
    
        
        if self.args.use_persona:
            persona =  raw['ground_persona']        
            persona = "</s>".join(ast.literal_eval(persona))
            # print(persona)
            if persona is None:
                persona = " "
        return self.tokenizer(knowledge=knowledge, question=question, persona=persona, history = history), self.tokenizer(answer=answer)


class FocusDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.args = args 
        self.tokenizer = Tokenizer(args)
    
    def prepare_data(self):
        self.train_dataset = FocusDataset(self.args, train=True)
        self.test_dataset = FocusDataset(self.args, train=False)
        
    def setup(self, stage=None):
        val_length = int(self.args.val_test_split_factor * len(self.test_dataset))
        self.val_dataset, self.test_dataset = random_split(self.test_dataset, [val_length, len(self.test_dataset) - val_length])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=self.args.num_workers,
                          batch_size=self.batch_size,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          num_workers=self.args.num_workers,
                          batch_size=self.batch_size,
                          shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=self.args.num_workers,
                          batch_size=self.batch_size,
                          shuffle=False)


class FocusModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.automatic_optimization = False
        self.args = args
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = Evaluator(args, self.device_)
        # self.generator = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        # self.tokenizer = GPT2Tokenizer.from_pretrained(
        #     "EleutherAI/gpt-neo-1.3B",    
        #     bos_token='<|startoftext|>',
        #     eos_token='<|endoftext|>',
        #     pad_token='<|pad|>')
        # self.generator.resize_token_embeddings(len(self.tokenizer))
        
        self.generator = BartForConditionalGeneration.from_pretrained(args.model_generator)
        self.tokenizer = BartTokenizer.from_pretrained(args.model_generator)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.gamma = args.gamma
        self.lr = args.learning_rate
        self.validation_step_outputs = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler' : torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_scheduler_step, gamma=self.args.lr_scheduler_factor),
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
        persona_batch = batch[0][2][0]
        # print("Persona Batch:", persona_batch)
        
        decoder_input_ids, decoder_attention_mask = batch[1][0].squeeze(1), batch[1][1].squeeze(1)
        out = self.generator(input_ids=input_ids, attention_mask=attention_mask, 
                            #  decoder_attention_mask=decoder_attention_mask,
                             labels=decoder_input_ids)
        loss, logits = out.loss, out.logits
        prob = torch.nn.functional.softmax(logits, dim=-1)
        batch_size = input_ids.shape[0]
        loss_tune = torch.zeros(batch_size)
        
        for i in range(batch_size):
            prediction = self.tokenizer.decode(prob[i, :, :].argmax(dim=-1).tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            ground_truth = self.tokenizer.decode(decoder_input_ids[i, :].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if self.args.use_persona:
                persona = persona_batch[i]
            else:
                persona = " "
            # persona = self.tokenizer.decode(input_ids[i, self.args.max_len_context-2:self.args.max_len_context+64].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # print(persona)
            r = 1 - self.evaluator(prediction, ground_truth, persona)
            loss_tune[i] = r * self.cross_entropy_loss(logits[i], decoder_input_ids[i])
        
        loss_tune = loss_tune.mean()
        loss = loss * self.gamma + (1 - self.gamma) * loss_tune
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log('val_loss', loss, on_step=True)
        self.validation_step_outputs.append(loss)
        return loss
    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", epoch_average)
        self.validation_step_outputs.clear()  # free memory
    
    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack(outputs).mean()
    #     self.log('ptl/val_loss', avg_loss)

    
class Evaluator:
    def __init__(self, args, device):
        self.args = args
        self.tokenizer = ElectraTokenizer.from_pretrained(args.model_evaluator)
        self.model = ElectraModel.from_pretrained(args.model_evaluator).eval().to(device)
        
        if self.args.use_persona:
            self.alpha = 0
            self.beta = 0
            self.delta = 1
        else:
            self.alpha = 0.5
            self.beta = 0.5
            self.delta = 0.0
            
        self.device = device
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def scale_reward(self, reward):
        return (reward + 1 - (self.alpha+self.beta + self.delta)) / (2 - (self.alpha+self.beta + self.delta))
        
    def __call__(self, sen1, sen2, persona):
        with torch.no_grad():
            input_ids = self.tokenizer([sen1.lower(), sen2.lower(), persona.lower()], return_tensors='pt', truncation=True, padding=True).to(self.device)
            out = self.model(**input_ids)['last_hidden_state']
            sim = self.cos(out[0, 0, :], out[1, 0, :]).item()
            
            persona_sim = 0
            if self.args.use_persona:
                persona_sim = self.cos(out[0, 0, :], out[2, 0, :]).item()
                
            sen1_tokens = word_tokenize(sen1)
            sen2_tokens = word_tokenize(sen2)
            bleu = sentence_bleu([sen2_tokens], sen1_tokens)
            # print(f'BLEU: {bleu}, SIM: {sim}, PERSONA_SIM: {persona_sim}')
            reward = self.alpha * bleu + self.beta * sim + self.delta * persona_sim
            return self.scale_reward(reward)
        

if __name__ == '__main__':
    train_df = pd.read_csv("/work/kanakr/chat_persona/data/dataset/train_data.csv")
    print(train_df.columns)
    # dataset = FocusDataset(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # train_df = train_df.iloc[:100]    
    # persona_token_counts, knowledge_token_counts, question_token_counts, answer_token_counts, history_token_counts = [], [], [], [], []
    
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # for _, row in train_df.iterrows():
    #     persona = " ".join(ast.literal_eval(row['ground_persona']))
    #     persona_token_count = len(tokenizer.encode(persona))
    #     persona_token_counts.append(persona_token_count)
        
    #     knowledge_token_count = len(tokenizer.encode(row['ground_knowledge']))
    #     knowledge_token_counts.append(knowledge_token_count)
        
    #     question_token_count = len(tokenizer.encode(row['question_rewritten']))
    #     question_token_counts.append(question_token_count)
        
    #     answer_token_count = len(tokenizer.encode(row['answer']))
    #     answer_token_counts.append(answer_token_count)
        
        
    #     history = ast.literal_eval(row['dialog_history'])
    #     history_size = min (2*2,  len(history)) 
    #     history = history[-history_size:]
    #     history_token_count = len(tokenizer.encode(" ".join(history)))
    #     history_token_counts.append(history_token_count)

    # fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(30, 5))
    # sns.histplot(persona_token_counts, ax=ax1)
    # ax1.set_title('Persona')

    # sns.histplot(knowledge_token_counts, ax=ax2)
    # ax2.set_title('Knowledge')

    # sns.histplot(question_token_counts, ax=ax3)
    # ax3.set_title('Question')

    # sns.histplot(answer_token_counts, ax=ax4)
    # ax4.set_title('Answer')
    
    # sns.histplot(history_token_counts, ax=ax5)
    # ax5.set_title('History')

    
    # assert len(persona_token_counts) == len(knowledge_token_counts) == len(question_token_counts) == len(answer_token_counts) == len(history_token_counts)
    # zipped_list = zip(persona_token_counts, knowledge_token_counts, question_token_counts, answer_token_counts, history_token_counts)
    # total_counts = [sum(item) for item in zipped_list]
    # sns.histplot(total_counts, ax=ax6)
    # ax6.set_title('Total')
    
    # plt.savefig('token_counts.png')
        
