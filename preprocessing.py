import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
"""
A function that cleanes possible errors like missing error's, unneccesary spaces or Capitalization
"""

def preprocessing(df):

    # if a ticket not include any important information the ticket gets droped, single missing values gets filled up with empty values
    df = df.dropna(subset=['title', 'description'], how = 'all')

    # as preparation for SentenceTransformer the title and the description gonna connect to 'content'
    df['content'] = df['title'].fillna("") + " " + df['description'].fillna("")

    # delete unneccesary spaces
    df['content'] = df['content'].astype(str).str.strip()

    # vectorize the content with the SentenceTransformer
    X_embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode(df['content'].tolist(), show_progress_bar=True)
    X_norm = normalize(X_embeddings)

    return X_norm