from models import *
from tqdm import tqdm
import json
from torchmetrics.functional import bleu_score
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.functional.text.bert import bert_score
from sys import exit
# declaringa a class
class obj:
      
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)
   
def dict2obj(dict1):
      
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)

def generate(model, tokenizer, knowledge, question, persona, device):
    input_ids, attention_mask = tokenizer(knowledge=knowledge, question=question, persona=persona)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    with torch.no_grad():
        predictions = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5, max_length=100, early_stopping=True)
        predictions = [predictions[0].tolist()]
        predictions = tokenizer.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    return predictions

if __name__ == '__main__':
    args = open('config.json').read()
    args = json.loads(args)
    args = dict2obj(args)
    checkpoint_path = "/work/kanakr/chat_persona/generator/t5_lightning/neural_qa/saved_weights/checkpoint-epoch=00-val_loss=nan.ckpt"
    device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(args)
    model = LitQGModel.load_from_checkpoint(checkpoint_path, args=args).eval().to(device)
    # data =  pd.read_csv("../../../data/focus_test_data.csv")
    data = pd.read_csv("/work/kanakr/chat_persona/t5_canard/test_question_rewritten_1.csv")
    data = data.iloc[200:1000]
    # print(data.columns)
    # exit()
    outputs = []
    idx = 0
    for row in tqdm(list(data.iterrows())):
        idx+=1
        # print(row)
        knowledge, question, persona, answer = row[1]['ground_knowledge'], row[1]['question_rewritten'], row[1]['ground_persona'], row[1]['answer']
        persona = " ".join(ast.literal_eval(persona))
        if persona is None:
          persona = " "
        predictions = generate(model = model.generator, tokenizer = tokenizer, knowledge = knowledge, question = question, persona=persona, device=device)
        outputs.append([row[1]['question_rewritten'], row[1]['ground_knowledge'], predictions[0] if type(predictions) is list else predictions])
        if idx%20==0:
            print("Query>>>>", question)
            print("Knowledge>>>>", knowledge)
            print("Ground>>>>", answer)
            print("Prediction>>>>", predictions[0] if type(predictions) is list else predictions)
            print("===============================================================")
            
    outputs = pd.DataFrame(data=outputs, columns=['query', 'knowledge', 'prediction'])
    outputs['answer'] = data['answer']
    # outputs.to_csv('./neural_predictions_q_rewritten_1.csv', index=False)
    # df = pd.read_csv('./neural_predictions_q_rewritten_1.csv')
    ground = list(data['answer'])
    preds = list(outputs['prediction'])
    
    bleu = bleu_score(preds, ground)
    print(bleu)
    # bert = bert_score(preds, ground)
    # score = {k: sum(vv)/len(vv) for k, vv in bert.items()}
    # print(score)