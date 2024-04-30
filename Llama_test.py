from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests

def read_problems(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = []
        prompts = []
        labels = []
        for line in file:
            if '", 0' in line:
                prompt, _ = line.strip().split('", 0')
                label = 0
            else:
                prompt, _ = line.strip().split('", 1')
                label = 1
            data.append({'prompt': prompt.lstrip('"'), 'label': int(label)})
            prompts.append(prompt.lstrip('"'))
            labels.append(int(label))
    return data, prompts, labels


# Evaluate the model
def predict_matrix(output, labels):
    output_ = [0 if r.strip().lower() == 'negative' else 1 for r in output]

    # Initialize counters
    TP, TN, FP, FN = 0, 0, 0, 0

    # Iterate over predictions and actual labels to update counters
    for o, l in zip(output_, labels):
        if o == 1 and l == 1:
            TP += 1
        elif o == 0 and l == 0:
            TN += 1
        elif o == 1 and l == 0:
            FP += 1
        elif o == 0 and l == 1:
            FN += 1

    # Calculate precision
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0

    # Return a dictionary with the metrics
    return {
        'Accuracy': accuracy,
        'True Positives': TP,
        'True Negatives': TN,
        'False Positives': FP,
        'False Negatives': FN
    }


def query(payload, headers, API_URL):
    response = requests.post(API_URL, headers=headers, json=payload)
    response = response.json()
    sentiment = response[0]['generated_text'].replace(" ", "")
    sentiment = sentiment.strip().lower().split("return_output")[1]
    if "sentiment=negative" in sentiment:
        return 0
    elif "sentiment=positive" in sentiment:
        return 1
    else:
        N_n = sentiment.count("negative")
        N_p = sentiment.count("positive")
        if N_n > N_p:
            return 0
        elif N_n == N_p:
            return -1
        else:
            return 1


if __name__ == '__main__':
    path = 'data/attack_files/imdb_test_Openai_attacked.txt'
    data, prompt, label = read_problems(path)

    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {"Authorization": "Bearer hf_"} # Add your own API key here

    outputs = []
    for i in range(len(prompt)):
        output = query({
            "inputs": f"review: {prompt[i]} sentiment: Please classify this review for 'positive' or 'negative' "
                      f"sentiment. return_output: sentiment=""",
        }, headers, API_URL)

        outputs.append(output)

    # save the output
    output_path = path.replace('.txt', '_Llama_output.txt')
    with open(output_path, 'w', encoding='utf-8') as file:
        for i in range(len(outputs)):
            file.write(f'{outputs[i]}, ')
    print(len(outputs))

    # calculate the metrics
    content = ''
    with open(output_path, 'r') as file:
        content = file.read()

    TextFooler_output = content[:-2].split(",")
    print(predict_matrix(TextFooler_output, label))
