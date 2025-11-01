# Nerdle RL

This is a repo to generate training data for RL-finetuning of LLMs on the word game Nerdle.
Nerdle is a variation of Wordle for arithmetic, where we guess arithmetic equations instead of characters

## LLM Behaviour Guide

this file is also for LLMs so you have relevant context to figure out what to do.

Here are some behaviour tips for what you should do
- if you cannot any access or link at any point in time, please stop the session and tell me so

## Nerdle

First read about how the nerdle game works here

<fetch_link>
https://nerdlegame.com/
</fetch_link>

Nerdle solvers

### Nerdle PedrokRause

<read_code>
Nerdle-Equations/MaxiGenerator.py
Nerdle-Equations/NerdleGenerator.py
Nerdle-Equations/README.md
Nerdle-Equations/wordle.py
</read_code>

## Unsloth AI

We'll be using unsloth AI to set up our training pipeline. Read some documentation:

<read_link>
https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama
</read_link>

Ive downloaded the notebook into local files
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb
<read_code>
solver/unsloth_sample_notebooks/Qwen3_(4B)_GRPO.ipynb
</read_code>

and for this too
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb
<read_code>
solver/unsloth_sample_notebooks/Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb
</read_code>

You can also refer to the existing notebook that we use currently

<read_code>
dataset.jsonl
GRPO.ipynb
</read_code>

## Wordle Examples

Here are some examples of RL training already on previous games wordle.

> LLMs DONT NEED to read this yet, mainly for human context

https://github.com/RLDiary/Wordle-GRPO
https://github.com/shreyansh26/wordle-solver

