from models import *
import json

   
# declaringa a class
class obj:
      
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)
   
def dict2obj(dict1):
      
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)

if __name__ == '__main__':
    args = open('config.json').read()
    args = json.loads(args)
    args = dict2obj(args)
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3, strict=False, verbose=True, mode='min')
    model_checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath="./saved_weights", filename='checkpoint-{epoch:02d}-{val_loss:.2f}', save_top_k=3, mode='min')
    
    trainer_args = {
        'accelerator': args.accelerator,
        'devices' : args.devices,
        'max_epochs' : args.max_epoch,
        'val_check_interval' : args.val_check_interval,
        'precision' : args.precision,
        'limit_train_batches': args.limit_train_batches,
        'limit_val_batches': args.limit_val_batches,
        'fast_dev_run' : args.fast_dev_run,
        # 'strategy': args.strategy,
                    }

    trainer = pl.Trainer(**trainer_args, callbacks=[early_stop_callback, model_checkpoint_callback])
    
    model = LitQGModel(args)
    if args.load_from_checkpoint:
        model.load_from_checkpoint(checkpoint_path = args.checkpoint_path, args = args)
    dm = SquadDataModule(args)
    trainer.fit(model, dm)