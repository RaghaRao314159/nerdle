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

> These instructions assume you are running from the parent folder so you can view multipe repos at once. I have cloned all these repos locally so you can refer to them

## Nerdle PedrokRause

https://github.com/pedrokkrause/Nerdle-Equations

<read_code>
Nerdle-Equations/MaxiGenerator.py
Nerdle-Equations/NerdleGenerator.py
Nerdle-Equations/README.md
Nerdle-Equations/wordle.py
</read_code>

## Nerdle Data Analysis

https://github.com/lfhohmann/nerdle-data-analysis

<read_code>
nerdle-data-analysis/0.equations_generator.ipynb
nerdle-data-analysis/2.simulator_mini_nerdle_random.ipynb
<read_code>

## Nerdle Solver

https://github.com/starypatyk/nerdle-solver

<read_code>
nerdle-solver/libnerdle.py
nerdle-solver/gen_perms.py
<read_code>

## Unsloth AI

We'll be using unsloth AI to set up our training pipeline. Here's some example notebooks:

<read_link>
https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama
</read_link>

for this notebook read

https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb
<read_code>
nerdle/unsloth_sample_notebooks/Qwen3_(4B)_GRPO.ipynb
</read_code>

for this notebook read
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb
<read_code>
nerdle/unsloth_sample_notebooks/Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb
</read_code>

You can also refer to the existing notebook that we use currently

<read_code>
nerdle/dataset.jsonl
nerdle/GRPO.ipynb
</read_code>

## Wordle Examples

Here are some examples of RL training already on previous games wordle

<fetch_link>
https://github.com/RLDiary/Wordle-GRPO
https://github.com/shreyansh26/wordle-solver
</fetch_link>

