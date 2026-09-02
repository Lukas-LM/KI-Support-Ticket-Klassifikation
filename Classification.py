from preprocessing import preprocessing
import joblib
from sqlalchemy import create_engine
from functions import load_config, current_date
import pandas as pd
from datetime import datetime
import requests
"""
The main function, it takes the best model and the support tickets, cleans them and predict the cluster
Args:
- df: The dataset what includes the support tickets
"""

def Classification():

    classifier = joblib.load('support_ticket_classifier.pkl')
    config = load_config()

    api_url = config['jira_settings']['api_url']
    project_key = config['jira_settings']['project_key']
    number_tickets = config['jira_settings']['max_results']
    
    db_name = config['database_settings']['db_name']
    db_type = config['database_settings']['db_type']
    table_facts = config['database_settings']['table_fact']
    
    # Fetch data from the API
    url = api_url
    start_at = 10001
    params = {
                   'jql': f'project = {project_key} ORDER BY key ASC',
                   'startAt': start_at,
                   'maxResults': number_tickets
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
          data = response.json()
          df_block = pd.json_normalize(data['issues'])
    
    df = pd.DataFrame({
               'id': df_block['key'],
               'title': df_block['fields.summary'],
               'description': df_block['fields.description'],
               'created_date': df_block['fields.created'],
               'status': df_block['fields.status.name'],
               'priority': df_block['fields.priority.name']
    })
    # Convert the 'created_date' column to datetime format and handle any errors
    # Change the 'created_date' to the current date and time
    df['created_date'] = pd.to_datetime(df['created_date'], format='%Y-%m-%dT%H:%M:%S.%f%z', errors='coerce')
    df['created_date'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    engine = create_engine(f"{db_type}{db_name}")

    df = preprocessing(df)

    df['cluster'] = classifier.predict(df['content'])

    for cluster in df['cluster'].unique():
        df_cluster = df[df['cluster'] == cluster]
        df_cluster.to_sql(
            name = f'{cluster}',
            con = engine,
            if_exists = 'append',
            index = False
        )