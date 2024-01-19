import json
import argparse
from models import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import BartTokenizer, GPT2Tokenizer

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)

def load_config(config_path):
    with open(config_path, 'r') as file:
        args = json.load(file)
    return dict2obj(args)

def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def initialize_tokenizer(tokenizer_type):
    if tokenizer_type == 'bart':
        return BartTokenizer.from_pretrained('facebook/bart-large')
    elif tokenizer_type == 'gpt2':
        return GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", bos_token='', eos_token='', pad_token='')
    else:
        raise ValueError("Unsupported tokenizer type")

def debug_dataset(tokenizer_type, args):
    tokenizer = initialize_tokenizer(tokenizer_type)
    train_dataset = FocusDataset(args, train=True)
    for idx, data in enumerate(train_dataset):
        (input_ids, attention_mask, persona), (labels, decoder_attention_mask) = data
        print([tokenizer.decode(input_id, skip_special_tokens=True) for input_id in input_ids])
        print()
        if idx >= 5:
            break

def initialize_trainer(args):
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3, strict=False, min_delta=0.001, verbose=True, mode='min')
    model_checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath="./saved_weights", filename='bart_0_0_1-checkpoint-{epoch:02d}-{val_loss:.3f}', save_top_k=1, mode='min')

    trainer_args = {
        'accelerator': args.accelerator,
        'devices': args.devices,
        'max_epochs': args.max_epochs,
        'val_check_interval': args.val_check_interval,
        'precision': args.precision,
        'limit_train_batches': args.limit_train_batches,
        'limit_val_batches': args.limit_val_batches,
        'fast_dev_run': True,
    }
    return pl.Trainer(**trainer_args, callbacks=[early_stop_callback, model_checkpoint_callback])

def train_model(trainer, model, dm):
    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--tokenizer', type=str, choices=['bart', 'gpt2'], help='Tokenizer type for debugging')
    args = parser.parse_args()

    config = load_config('config.json')
    model = FocusModel(config)

    if config.load_from_checkpoint:
        model.load_from_checkpoint(checkpoint_path=config.checkpoint_path, args=config)

    if args.debug:
        debug_dataset(args.tokenizer, config)
    else:
        dm = FocusDataModule(config)
        trainer = initialize_trainer(config)
        train_model(trainer, model, dm)