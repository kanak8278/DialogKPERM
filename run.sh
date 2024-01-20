# # create and activate conda environment
conda env create -f environment.yml
conda activate my_env

# # Question Rewriting
# cd question_rewriting/
# python main.py


# # Training Baseline FoCus-BART Model
# cd generator/baseline/
# python train.py #Without Debug Mode
# python train.py --debug --tokenizer bart #Debug Dataset 
# python test.py


# # Running K-PERL
# cd generator/model_feedback/
# python train.py # training
# python test.py # testing

# # Evaluating with Nubia Score
# cd generator/
# python nubiaScore.py /path/to/your/file.csv/
