# EvasionAttackLLMs
A project that demonstrates Evasion Attack on LLMs

## Attack Methods
This project explores two primary methods of evasion attacks:

- **White-Box Attack**: SALSA attack on BERT. For more details, refer to the paper available at [SALSA: Salience-Based Switching Attack for Adversarial Perturbations in Fake News Detection Models](https://doi.org/10.1007/978-3-031-56069-9_3).
- **Black-Box Attack**: Prompt-based attack by ChatGPT. Further information can be found on ArXiv at [An LLM can Fool Itself: A Prompt-Based Adversarial Attack](https://arxiv.org/abs/2310.13345).

## Threat Models
The project specifically targets the following large language models as potential threat models:

1. **[BERT](https://huggingface.co/textattack/bert-base-uncased-yelp-polarity)**: A transformer-based model that excels in understanding the context within natural language.
2. **[Llama-3-8B](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct)**: A variant of large language models optimized for specific tasks with an 8 billion parameter configuration.
3. **[ChatGPT](https://platform.openai.com/docs/models/gpt-3-5-turbo)**: Based on the GPT architecture, designed to generate human-like text in response to prompts.

