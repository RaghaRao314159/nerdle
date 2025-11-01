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

Scrape through all these codebases using your github tools to determine whats the optimum library to use (most comprehensive and well written code)

<fetch_link>
https://github.com/pedrokkrause/Nerdle-Equations
https://github.com/lfhohmann/nerdle-data-analysis
https://github.com/starypatyk/nerdle-solver
</fetch_link>


## Unsloth AI

We'll be using unsloth AI to set up our training pipeline. Here's an example notebook

<fetch_link>
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb
https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb
</fetch_link>

You can also refer to the existing notebook that we use currently

<read_code>
GRPO.ipynb
dataset.jsonl
</read_code>

## Wordle Examples

Here are some examples of RL training already on previous games wordle

<fetch_link>
https://github.com/RLDiary/Wordle-GRPO
https://github.com/shreyansh26/wordle-solver
</fetch_link>


