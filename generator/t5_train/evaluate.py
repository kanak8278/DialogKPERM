import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from utils import DialogDataset, validate

args = {
    # Initialize config
    "TRAIN_BATCH_SIZE" : 16,    # input batch size for training (default: 64)
    "VALID_BATCH_SIZE" : 16,    # input batch size for testing (default: 1000)
    "TRAIN_EPOCHS" : 40,        # number of epochs to train (default: 10)
    "VAL_EPOCHS" : 1, 
    "LEARNING_RATE" : 5e-5,    # learning rate (default: 0.01)
    "SEED" : 42,               # random seed (default: 42)
    "MAX_LEN" : 512,
    "ANSWER_LEN" : 250, 
    "MODEL_SAVE_DIR" : "t5_grd_knw_grd_persona/"

}

val_params = {
    'batch_size': args["VALID_BATCH_SIZE"],
    'shuffle': True,
    'num_workers': 4
    }


# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(args["SEED"]) # pytorch random seed
np.random.seed(args["SEED"]) # numpy random seed
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained(args["MODEL_SAVE_DIR"])
    print("Inference Model Ready")
    
    test = "../data/focus_test_data.csv"
    test_df = pd.read_csv(test)
    print("TEST Dataset: {}".format(test_df.shape))
    test_dataloader = DataLoader(DialogDataset(tokenizer, test_df, args["MAX_LEN"], args["ANSWER_LEN"]), **val_params)
    print("Test Mini-Batch: ", len(test_dataloader))

    print("Evaluation starts on Test Data:")
    predictions, actuals = validate(tokenizer, model, test_dataloader)
    print("Predictions Ready!")
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
    final_df.to_csv(f't5_predictions.csv', index=False)
    print('Output Files generated for review')

    