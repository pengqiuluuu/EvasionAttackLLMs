# Evasion Attack on LLMs
A project that demonstrates Evasion Attack on LLMs in sentiment analysis tasks.

## Attack Methods
This project explores two primary methods of evasion attacks:

- **White-Box Attack**: SALSA attack on BERT. For more details, refer to the paper available at [SALSA: Salience-Based Switching Attack for Adversarial Perturbations in Fake News Detection Models](https://doi.org/10.1007/978-3-031-56069-9_3).
- **Black-Box Attack**: Prompt-based attack by ChatGPT. Further information can be found on ArXiv at [An LLM can Fool Itself: A Prompt-Based Adversarial Attack](https://arxiv.org/abs/2310.13345).

## Threat Models
The project specifically targets the following large language models as potential threat models:

1. **[BERT](https://huggingface.co/textattack/bert-base-uncased-yelp-polarity)**: A transformer-based model that excels in understanding the context within natural language.
2. **[Llama-3-8B](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct)**: A variant of large language models optimized for specific tasks with an 8 billion parameter configuration.
3. **[ChatGPT](https://platform.openai.com/docs/models/gpt-3-5-turbo)**: Based on the GPT architecture, designed to generate human-like text in response to prompts.

## Sentiment analysis task
- Dataset: [IMDb dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
- labels: Either positive or negative
- Size: For this experiment, we only take a subset of 1000 reviews to demonstrate and generate our adversarial examples 

## Project Presentation
- Youtube：https://youtu.be/EH1s5jgB8Qc
- Slides: [Link](https://docs.google.com/presentation/d/1nizdzbujQit8zoSiRoU-ZDNBRCoH4meS/edit?usp=sharing&ouid=104701225868345278149&rtpof=true&sd=true)
