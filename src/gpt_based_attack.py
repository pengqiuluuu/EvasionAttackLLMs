from tqdm import tqdm
import sys
import openai
from openai import OpenAI
from typing import List, Optional
OPENAI_API_KEY=''


def read_problems(path):
    with open(path, 'r', encoding='utf-8') as file:
        contents = []
        labels = []
        prompts = []
        for line in file.readlines():
            if '", 0' in line:
                content, _ = line.strip().split('", 0')
                label = 0
            else:
                content, _ = line.strip().split('", 1')
                label = 1
            contents.append(content.lstrip('"'))
            labels.append(int(label))

            if label == 0:
                prompt = f"Try to perturb the following review to positive sentiment: \n{contents[-1]}"
            else:
                prompt = f"Try to perturb the following review to negative sentiment: \n{contents[-1]}"
            prompts.append(prompt)

    return contents, labels, prompts


class OpenAIChatDecoder:
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__()
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.name = name

    def __call__(self, adv_prompt) -> str:
        completion = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": "Try to change the sentiment of given review within 10 imperceptible "
                                              "perturbations. Return the perturbed review only."},
                {"role": "user", "content": adv_prompt}
            ]
        )
        adv_output = completion.choices[0].message.content
        return adv_output



if __name__ == '__main__':
    # load the text file
    path = 'imdb_test.txt'
    output_path = 'result/' + path.replace('.txt', f'_Openai_attacked.txt')
    content, labels, prompts = read_problems(path)

    # construct the attack prompt
    perturbation_instructions =[
        "Choose at most two words in the sentence, and change them so that they have typos.",
        "Change at most two letters in the sentence.",
        "Add at most two extraneous characters to the end of the sentence.",
        "Replace at most two words in the sentence with synonyms.",
        "Choose at most two words in the sentence that do not contribute to the meaning of the sentence and delete them.",
        "Add at most two semantically neutral words to the sentence.",
        "Add a randomly generated short meaningless handle after the entence, such as @fasuv3.",
        "Paraphrase the sentence.",
        "Change the syntactic structure of the sentence.",
    ]


    # define the model
    name = 'gpt-3.5-turbo'
    pipeline = OpenAIChatDecoder(name=name)

    # attack the text
    for prompt, label in tqdm(zip(prompts, labels), total=len(prompts), file=sys.stdout):
        prompt = f"{prompt} \nUse the following instructions: \n{perturbation_instructions}"

        output = pipeline(prompt)
        with open(output_path, 'a', encoding='utf-8') as file:
            line = output.replace('\n', ' ')
            file.write(f'"{line}", {label}\n')
    print(f'File saved at {output_path}')



