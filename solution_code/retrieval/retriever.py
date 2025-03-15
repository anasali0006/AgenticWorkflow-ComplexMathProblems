import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import get_embedding


class RetrievalEngine:
    """
    This class implements retrieval engine, which performs two steps filtering of the data. 
    1. It filters the data based on the abbreviation of the organization
    2. It narrows downs the results within an organization based on semantic similarity and lexical similarity

    For this prototype, I am using pandas dataframe to keep it simple, 
    but for production application, a vector database will be used
    """

    def __init__(self):

        self.top_k = 5

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.PROJECT_DIR = os.path.dirname(self.BASE_DIR)
        self.data_with_semantic_vectors_path = os.path.join(self.PROJECT_DIR, "data", "dense_embedddings_with_index.parquet") 
        self.data_with_tfidf_vectors_path = os.path.join(self.PROJECT_DIR, "data", "sparse_embeddings_with_index.pkl") 
        self.vectors_df = self.load_vector_data()

        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer.fit(self.vectors_df["Context"].tolist())
        self.embedding_model = "text-embedding-3-large"


    def load_vector_data(self):
        # Load the data from files, which contain pre-computed vectors for our data
        # These vectors are created using OpenAI text_embeddings_3_large model
        # The lexical vectors are created using TF-IDF vectorizer from Scipy

        semantic_df = pd.read_parquet(self.data_with_semantic_vectors_path)
        tfidf_df = pd.read_pickle(self.data_with_tfidf_vectors_path)
        df_combined = pd.concat([semantic_df, tfidf_df], axis=1)
        df_combined["Abbreviation"] = df_combined.index.str.split("/").str[0].str.split("_").str[-1]
        df_combined["Combined_Vector"] = df_combined.apply(self.combine_vectors, axis=1)

        return df_combined


    def combine_vectors(self, row):
        # Function to combine contextual and lexical embeddings in dataframe
        tfidf_dense = row["Context_TFIDF_Vector"].toarray().flatten()
        embedding = np.array(row["embedding_context"])
        combined_vector = np.concatenate([embedding, tfidf_dense])
        return combined_vector



    def filter_organizations(self, organization_abbreviation):
        # This method filters the given documents, and selects the ones which are 
        # related to the organization user asked. 
        selected_df = self.vectors_df[self.vectors_df["Abbreviation"] == organization_abbreviation]
        return selected_df
    

    def filter_within_organziation(self, client, user_query, org_df):
        # This method applies hybrid search (semantic + lexical) search to select top K documents
        semantic_embedding_user_query = get_embedding(client, user_query, self.embedding_model)
        td_idf_embedding_user_query = self.tfidf_vectorizer.transform([user_query])

        combined_embedding_user_query = np.hstack([semantic_embedding_user_query, td_idf_embedding_user_query.toarray().flatten()])

        matrix = np.vstack(org_df["Combined_Vector"].values)
        query_vector = np.array(combined_embedding_user_query).reshape(1, -1)
        similarities = cosine_similarity(matrix, query_vector).flatten()
        top_k_indices = np.argsort(similarities)[-self.top_k:][::-1]
        final_df = org_df.iloc[top_k_indices]

        return final_df


    def run_retrieval_engine(self, client, user_query, organization_abbreviation):
        # This method runs the retrieval engine with steps. 
        org_df = self.filter_organizations(organization_abbreviation)
        org_df = self.filter_within_organziation(client, user_query, org_df)
        
        return org_df[["Context", "Question", "Dialogue"]]
