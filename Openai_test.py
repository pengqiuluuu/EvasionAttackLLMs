import openai
import os
from openai import OpenAI
from typing import List, Optional

os.environ.get("OPENAI_API_KEY")
OPENAI_API_KEY = 'sk-'


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


class OpenAIChatDecoder:
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__()
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def test(self, prompt: List[str], num_samples: int = 200) -> list[Optional[str]]:
        outputs = []
        for i in range(len(prompt)):
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "Please classify the following sentence as a positive or negative sentiment. Return 'positive' or 'negative' only."},
                    {"role": "user", "content": prompt[i]}
                ]
            )
            outputs.append(completion.choices[0].message.content)

        return outputs


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


if __name__ == '__main__':
    path = 'data/attack_files/imdb_test_Openai_attacked.txt'
    data, prompts, labels = read_problems(path)

    name = 'GPT-3.5-turbo'

    # Predict the sentiment
    pipeline = OpenAIChatDecoder(name=name)
    output = pipeline.test(prompts)

    output_path = path.replace('.txt', '_Openai_output.txt')

    with open(output_path, 'w', encoding='utf-8') as file:
        for i in range(len(output)):
            file.write(f'{output[i]}, ')
    print(len(output))

    content = ''
    with open(output_path, 'r') as file:
        content = file.read()

    TextFooler_output = content[:-2].split(",")
    print(predict_matrix(TextFooler_output, labels))
