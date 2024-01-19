#exp2 takes modified hit knoweldge best of (query rewritten and query)
#exp3 takes history in account

from models import *
from tqdm import tqdm
import json
from torchmetrics.functional import bleu_score
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.text.bert import BERTScore
# from nubia_score import Nubia

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
# from bert_score import BERTScorer 
# from rouge import Rouge
# cc = SmoothingFunction()
# scorer = BERTScorer(lang="en", rescale_with_baseline=True)

from sys import exit
from pprint import pprint

# declaring a class
class obj:
      
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)
   
def dict2obj(dict1):
      
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)

def generate(model, tokenizer, knowledge, question, persona, history, device):
    input_ids, attention_mask, persona_batch = tokenizer(knowledge=knowledge, question=question, persona=persona, history=history)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    with torch.no_grad():
        predictions = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5, max_length=100, early_stopping=True)
        predictions = [predictions[0].tolist()]
        predictions = tokenizer.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    return predictions

if __name__ == '__main__':
    
    # print("===================================================================================")
    # df = pd.read_csv("/home/ubuntu/chat_persona/data/focus_test_data.csv")
    # print(df.columns)
    # pprint(df.iloc[12].to_dict())
    # print("===================================================================================")
    
    # df = pd.read_csv("/home/ubuntu/chat_persona/data/question_rewritten/test_question_rewritten_hit_knowledge_1.csv")
    # print(df.columns)
    # pprint(df.iloc[12].to_dict())
    
    # df = pd.read_csv("/home/ubuntu/chat_persona/data/question_rewritten/test_question_rewritten_hit_knowledge_1.csv")
    # df_1 = pd.read_csv("/home/ubuntu/chat_persona/generator/bart_train/neural_predictions_q_rewritten_1.csv")
    # df_0 = pd.read_csv("/home/ubuntu/chat_persona/generator/bart_train/neural_predictions_q_rewritten_0.csv")
    # print(df.columns)
    # print(df.shape, df_0.shape, df_1.shape)
    
    # for idx, row in enumerate(zip(df.iterrows(), df_0.iterrows(), df_1.iterrows())):
    #     if idx >= 200:   
    #         row, row_0, row_1 = row[0][1], row[1][1], row[2][1]
    #         print("Query:", row['query'])
    #         print("Query_Rewritten:", row_0['query'])
    #         persona = " ".join(ast.literal_eval(row['ground_persona']))
    #         print("Persona:", persona)
    #         print("Hit Knowledge:", row_0['knowledge'])
    #         print("Prediction_0 ", row_0['prediction'])
    #         print("Prediction_1: ", row_1['prediction'])
    #         print("Ground Answer: ", row['answer'])
    #         print("===============================================================")
    #     if idx >= 500:
    #         break

    

    # saved_file = "./bart_predictions_exp_5_pred_persona.csv"

    # """Generation Code"""
    # df = pd.read_csv("/work/kanakr/chat_persona/data/dataset/test_data.csv")
    # print(df.columns, df.shape)
    # print("Predicted Persona File Loaded")
    # persona_test = pd.read_csv("/work/kanakr/chat_persona/data/test_persona_results.csv")
    # print(persona_test.columns, persona_test.shape)
    # print("===================================================================================")
    
    # args = open('config.json').read()
    # args = json.loads(args)
    # args = dict2obj(args)
    
    # print("Use Persona:", args.use_persona)
    # print("Use Knowledge:", args.use_knowledge)
    # print("Use History:", args.use_history)
    
    # df = pd.merge(df, persona_test[['dialogID', 'utterance', 'pred_persona']], on=['dialogID', 'utterance'])
    # print(df.columns, df.shape)
  
    
    # persona_col = "pred_persona"
        
    # checkpoint_path = "/work/kanakr/chat_persona/generator/bart_train/saved_weights/checkpoint-epoch=04-val_loss=0.372_all_persona.ckpt"
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokenizer = Tokenizer(args)
    # model = FocusModel.load_from_checkpoint(checkpoint_path, args=args).eval().to(device)

    # outputs = []
    # idx = 0
    # for row in tqdm(list(df.iterrows())):
    #     idx+=1

    #     # knowledge, question,  answer = row[1]['hit_knowledge'], row[1]['query'],  row[1]['answer']
    #     knowledge, history, persona = None, None, None
    #     question, answer  =  row[1]['query'], row[1]['answer']
        
    #     if args.use_knowledge:
    #         knowledge = row[1]['hit_knowledge']   
        
    #     if args.use_history:
    #         history = ast.literal_eval(row[1]['dialog_history'])
    #         if type(history) is not list or history is None or history == []:
    #             history = " "
    #         else:
    #             history_size = min (args.history_size,  len(history)) 
    #             history = history[-history_size:]
    #             history = " ".join(history)
    
        
    #     if args.use_persona:
    #         if persona_col == "pred_persona":
    #             persona = row[1]['pred_persona']
    #         else:    
    #             persona =  row[1]['ground_persona']        
    #             persona = "</s>".join(ast.literal_eval(persona))
    #             if persona is None:
    #                 persona = " "
            
                        
    #     predictions = generate(model = model.generator, tokenizer = tokenizer, knowledge = knowledge, question = question, persona=persona, history = history, device=device)
    #     outputs.append(
    #         {
    #             'dialogID': row[1]['dialogID'],
    #             'utterance': row[1]['utterance'],
    #             'prediction': predictions[0] if type(predictions) is list else predictions,
    #             'answer': answer
    #         }
    #     )
        
    #     # outputs.append([row[1]['query'], row[1]['hit_knowledge'], predictions[0] if type(predictions) is list else predictions])
        
    #     if idx%51==0:
    #         print("Query>>>>>>>>", question)
    #         print("Knowledge>>>>", row[1]['hit_knowledge'])
    #         print("Ground Persona>>>>>>", "</s>".join(ast.literal_eval(row[1]['ground_persona'])))
    #         print("Selected Persona>>>>>>", row[1]['pred_persona'])
    #         print("Ground>>>>>>>", answer)
    #         print("Prediction>>>", predictions[0] if type(predictions) is list else predictions)
    #         print("===============================================================")
          
    # outputs = pd.DataFrame(data=outputs)
    # # outputs['answer'] = df['answer']
    # outputs.to_csv(saved_file, index=False)
    file = "/work/kanakr/chat_persona/generator/bart_train/predictions/godel_test_result.csv"
    df = pd.read_csv(file)
    print(df.columns)
    # exit()
    df['response'] = df['response'].fillna(" ")
    
    ground = list(df['answer'])
    preds = list(df['response'])
    
    """Debug"""
    # print(df.head())
    
    # for idx, row in enumerate(df.iterrows()):
    #     print("Query: ", row[1]['query'])
    #     print("Knowledge: ",row[1]['hit_knowledge'])
    #     print("Prediction: ",row[1]['prediction'])
    #     print("Ground: ", row[1]['answer'])
    #     print("===============================================================")
    #     if idx >=50:
    #         break
    # nubia = Nubia()
    # df['prediction'] = df['prediction'].fillna(" ")
    # for grd, pred in zip(ground, preds):
    #     print("Ground: ", grd)
    #     print("Pred: ", pred)
    #     score = bleu_score([pred], [grd])
    #     print("Score: ", score)
    #     print("===============================================================")
    
    
    """Metrics Calculation"""
    print("Metrics evaluation starting:")
    print("===================================================================================")
    
    print("Calculating Bleu Score")
    bleu = bleu_score(preds[:], ground[:])
    print("Bleu Score: ", bleu)
    print("===================================================================================")
    
    print("Calculating Rouge Score")
    rouge = rouge_score(preds[:], ground[:])
    print("Rouge Score: ", rouge)
    print("===================================================================================")
    
    bertscore = BERTScore('bert-base-uncased')
    print("Calculating BERT Score")
    bert = bertscore(preds[:], ground[:])
    bert = {k: sum(vv)/len(vv) for k, vv in bert.items()}
    print("BERT Score: ", bert)
    print("===================================================================================")
   