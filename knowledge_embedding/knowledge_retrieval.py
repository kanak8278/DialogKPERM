import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import nltk
nltk.download('punkt')
from nltk import sent_tokenize

class KnowledgeRetriever:
    def __init__(self, bi_encoder_model, cross_encoder_model, similarity_model, top_k=32):
        self.bi_encoder = SentenceTransformer(bi_encoder_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.similarity_model = SentenceTransformer(similarity_model)
        self.bi_encoder.max_seq_length = 1024
        self.top_k = top_k

    def get_paragraphs(self, knowledge):
        return [sent_tokenize(paragraph.strip()) for paragraph in knowledge.split("\n") if paragraph.strip()]

    def get_passages(self, paragraphs, window_size=1):
        passages = []
        for paragraph in paragraphs:
            for start_idx in range(0, len(paragraph), window_size):
                end_idx = min(start_idx + window_size, len(paragraph))
                passages.append(" ".join(paragraph[start_idx:end_idx]))
        return passages

    def retrieve_knowledge(self, query, passages, corpus_embeddings, n_count=3):
      question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
      question_embedding = question_embedding.cuda()
      hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=self.top_k)
      hits = hits[0]  # Get the hits for the first query

      cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
      cross_scores = self.cross_encoder.predict(cross_inp)

      # Sort results by the cross-encoder scores
      for idx in range(len(cross_scores)):
          hits[idx]['cross-score'] = cross_scores[idx]

      hits = sorted(hits, key=lambda x: x['score'], reverse=True)

      hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
      hit_passages = []
      hit_cross_scores = []
      for hit in hits[0:n_count]:
        hit_cross_scores.append(hit['cross-score'])
        hit_passages.append(passages[hit['corpus_id']].replace("\n", " "))

      return hit_cross_scores, hit_passages

    def get_result(self, query, passages, corpus_embeddings, n_count=2):
        hit_scores, hit_passages = self.retrieve_knowledge(query, passages, corpus_embeddings, n_count=n_count)
        hit_knowledge = " ".join(hit_passages)
        embeddings1 = self.similarity_model.encode(ground_knowledge, convert_to_tensor=True)
        embeddings2 = self.similarity_model.encode(hit_knowledge, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        return hit_knowledge, cosine_scores[0].item()

    def process_dataframe(self, df, test_data):
        """
        Processes a dataframe and extracts knowledge based on the test data.
        Args:
            df (DataFrame): The pandas DataFrame containing dialogues and other information.
            test_data (list): A list of test data entries.

        Returns:
            list: A list of result dictionaries.
        """
        results = []
        last_dialogID = None
        dialogID_index = -1
        flag = False
        corpus_embeddings = None

        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            dialogID, query_rewritten, query, utterance, ground_knowledge = self._extract_row_data(row)
            if dialogID != last_dialogID:
                dialogID_index += 1
                last_dialogID = dialogID
                flag = True

            assert dialogID == test_data[dialogID_index]['dialogID']

            knowledge, persona_candidates, persona_grounding = self._get_test_data(test_data, dialogID_index, utterance)

            if flag:
                corpus_embeddings = self._encode_passages(knowledge)
                    # Generate passages from the knowledge
            paragraphs = self.get_paragraphs(knowledge)
            passages = self.get_passages(paragraphs, window_size=1)
            result = self._compare_queries(dialogID, utterance, query, query_rewritten, passages, corpus_embeddings, ground_knowledge, persona_candidates, persona_grounding)
            if result:
                results.append(result)

        return results

    def _extract_row_data(self, row):
        """Extracts necessary data from a DataFrame row."""
        return row['dialogID'], row['question_rewritten'], row['query'], row['utterance'], row['ground_knowledge']

    def _get_test_data(self, test_data, dialogID_index, utterance):
        """Retrieves and formats test data for a given dialog ID."""
        test_entry = test_data[dialogID_index]
        knowledge = "\n".join(test_entry['knowledge'])
        persona_candidates = test_entry['utterance'][utterance]['persona_candidate']
        persona_grounding = test_entry['utterance'][utterance]['persona_grounding']
        return knowledge, persona_candidates, persona_grounding

    def _encode_passages(self, knowledge):
        """Encodes the passages from knowledge."""
        paragraphs = self.get_paragraphs(knowledge)
        passages = self.get_passages(paragraphs, window_size=1)
        try:
            return self.bi_encoder.encode(passages, convert_to_tensor=True)
        except Exception as e:
            print(f"Encoding Error: DialogId: {dialogID}, Exception: {e}")
            return None

    def _compare_queries(self, dialogID, utterance, query, query_rewritten, passages, corpus_embeddings, ground_knowledge, persona_candidates, persona_grounding):
      """Compares the original and rewritten queries to find the best match."""
      if corpus_embeddings is None:
          return None

      try:
          hit_knowledge_0, score_0 = self.get_result(query, passages, corpus_embeddings, n_count=1)
          hit_knowledge_1, score_1 = self.get_result(query_rewritten, passages, corpus_embeddings, n_count=1)

          if score_0 > score_1:
              chosen_query, hit_knowledge, score = query, hit_knowledge_0, score_0
          else:
              chosen_query, hit_knowledge, score = query_rewritten, hit_knowledge_1, score_1

          return {
              "dialogID": dialogID,
              "utterance": utterance,
              "query": chosen_query,
              "hit_knowledge": hit_knowledge,
              "ground_knowledge": ground_knowledge,
              "persona_candidates": persona_candidates,
              "persona_grounding": persona_grounding,
              "similarity_score": score
          }
      except Exception as e:
          print(f"Comparison Error: DialogId: {dialogID}, Exception: {e}")
          return None


retriever = KnowledgeRetriever(
    bi_encoder_model='multi-qa-MiniLM-L6-cos-v1',
    cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-12-v2',
    similarity_model='all-MiniLM-L6-v2'
)

# Step 2: Load Data
df_path = "path/to/your_dataframe.csv"  # Update with your file path
test_data_path = "path/to/your_test_data.json"  # Update with your file path

df = pd.read_csv(df_path)
with open(test_data_path) as file:
    test_data = json.load(file)

results = retriever.process_dataframe(df, test_data)


result_df = pd.DataFrame(results)
result_df.to_csv("path/to/your_output.csv", index=False)


# *   Window-2 & Count-2 : 0.68034
# *   Window-2 & Count-1 : 0.68034
# *   Window-1 & Count-1 : 0.682617
# *   Window-1 & Count-2 : 0.682617
# *   Window-1 & Count-3 : 0.682617


# """

# scores = [result['similarity_score'] for result in results]
# sum(scores)/len(scores)

# sum(hit_knowledge_df['similarity_score'].values[:])/len(hit_knowledge_df['similarity_score'].values[:])

# result_df = pd.DataFrame(results)

# result_df.columns

# folder = "/content/drive/MyDrive/UMBC/knowledge_retrieval"

# result_df.to_csv(f"{folder}/train_query_rewritten_1_1.csv", index = False)

# index =15
# print("Query: ", results[index]['query'])

# print( results[index]['hit_knowledge'])

# print("Query: ", hit_knowledge_df.iloc[index]['query'])

# print(hit_knowledge_df.iloc[index]['hit_knowledge'])

# hit_knowledge_df.columns

# hit_knowledge_df.head()

# result_df.shape

