from transformers import BertTokenizer, BertForSequenceClassification
import torch


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


# Function to classify sentiment of a sentence
def classify_sentiment(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    classification = torch.argmax(probs, dim=1)
    return "Positive" if classification.item() == 1 else "Negative"


# calculate the metrics
def predict_matrix(output, labels):
    output_ = [0 if r.strip().lower() == 'negative' else 1 for r in output]

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


if __name__ == '__main__':
    path = 'attack/result/imdb_test_Openai_attacked.txt'
    data, prompt, label = read_problems(path)

    # Load the pre-trained BERT model and tokenizer
    model_name = "textattack/bert-base-uncased-yelp-polarity"
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # List of sentences
    sentences = prompt

    # Classify each sentence and write the output to a file
    output = [classify_sentiment(sentence, model, tokenizer) for sentence in sentences]
    output_path = path.replace('.txt', '_BERT_output.txt')
    with open(output_path, 'w', encoding='utf-8') as file:
        for i in range(len(output)):
            file.write(f'{output[i]}, ')

    # Read the output file and calculate the metrics
    content = ''
    with open(output_path, 'r') as file:
        content = file.read()

    TextFooler_output = content[:-2].split(",")
    len(TextFooler_output)
    predict_matrix(TextFooler_output, label)
