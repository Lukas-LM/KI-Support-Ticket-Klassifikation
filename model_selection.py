from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from preprocessing import preprocessing
from sqlalchemy import create_engine
import joblib
import pandas as pd
from functions import load_config
from create_data import data
from clusteringEmbedding import clustering

def model_selektion():

    config = load_config()

    db_name = config['database_settings']['db_name']
    db_type = config['database_settings']['db_type']
    table_facts = config['database_settings']['table_fact']

    engine = create_engine(f"{db_type}{db_name}")
    df = pd.read_sql(f"SELECT title, description, cluster FROM {table_facts}", con=engine)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['cluster'])

    X_train = preprocessing(train_df)
    X_test = preprocessing(test_df)

    y_train = train_df['cluster'].values
    y_test = test_df['cluster'].values

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    }
    best_clf = None
    best_f1 = -1.0

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred, average='weighted')
            print(f'{name}: {score}')

            if score > best_f1:
                best_f1 = score
                best_clf = model
        except Exception as e:
            print(f"Error in {name}: {str(e)}\n")

    if best_clf is not None:
        df = df.copy()
        df['content'] = df['title'].fillna('').astype(str) + ' ' + df['description'].fillna('').astype(str)
        X_full = preprocessing(df)
        y_full = df['cluster'].values
        best_clf.fit(X_full, y_full)
        try:
            joblib.dump(best_clf, "support_ticket_classifier.pkl")
        except Exception as e:
            print(f"Error in model selection: {str(e)}\n")

    return best_clf


if __name__ == "__main__":
    data()
    clustering()
    model_selektion()
    print("ready")