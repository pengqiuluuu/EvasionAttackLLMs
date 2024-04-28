from typing import OrderedDict
import csv
import re

def read_csv_dataset(file_path):
    data = []
    with open(file_path, mode='r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            review = str("row['text']") + ", " + row['label']
            data.append({"text": row['text'], "label": row['label']} )

    # save in txt file
    with open('random_200inject_test_data_salience_attack.txt', 'w', encoding='utf-8') as f:
        for item in data:
            review = item['text']
            label = item['label']
            line = f'"{review}", {label}'
            f.write(line + '\n')

    return data

# Usage
path = 'random_200inject_test_data_salience_attack.csv'
data = read_csv_dataset(path)

