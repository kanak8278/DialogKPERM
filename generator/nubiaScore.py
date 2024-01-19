from nubia_score import Nubia
from tqdm import tqdm
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    parent_path = path[:-4]
    return df, parent_path

def compute_nubia_scores(df, nubia_instance):
    results = []
    df_sampled = df.sample(frac=0.1, replace=True, random_state=1)
    for idx, (dialogID, utterance, pred, ans) in tqdm(enumerate(zip(df_sampled['dialogID'], df_sampled['utterance'], df_sampled['prediction'], df_sampled['answer']))):
        result = {"dialogID": dialogID, "utterance": utterance}
        try:
            score = nubia_instance.score(pred, ans, get_features=True)
            result['nubia_score'] = score['nubia_score']
            result.update(score['features'])
            results.append(result)
        except Exception as e:
            print(f"Index {idx}, {dialogID, utterance} has error!")
            print(e)
            print()
        if idx % 100:
            print(f"Index {idx} completed")
            break
    return pd.DataFrame(results)

def save_results(result_df, parent_path):
    result_df.to_csv(f"{parent_path}_nubia_score_temp.csv", index=False)
    nubia_s = list(result_df['nubia_score'])
    print("Nubia Score:", sum(nubia_s)/len(nubia_s))

if __name__ == '__main__':
    n = Nubia()
    paths = ["/work/kanakr/chat_persona/generator/bart_train/predictions/focus_inference_bart_base_20_LM.csv"]

    for path in paths:
        print("================================================================================================")
        print(path)
        df, parent_path = load_data(path)
        print(df.columns)
        
        result_df = compute_nubia_scores(df, n)
        save_results(result_df, parent_path)
        print()
