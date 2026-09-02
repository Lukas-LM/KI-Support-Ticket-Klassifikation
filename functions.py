import json
import os

def load_config(file_path='config.json'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Die Konfigurationsdatei '{file_path}' wurde nicht gefunden!")
    
    with open(f'{file_path}','r', encoding='utf-8') as f:
        config = json.load(f)

    return config

def save_config(config, file_path='config.json'):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
