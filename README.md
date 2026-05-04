# Building a Universal Game-Playing Agent Infrastructure to Optimize Performance

This repository supports the poster presentation of my Individual Module at the University of Potsdam. It extends [clemcore](https://github.com/clp-research/clemcore) — a framework for evaluating large language models by having them play Dialogue Games — with a set of custom agent architectures and experiments that explore how different agent architectures affect model performance across games.

> **Note:** This is a fork of `clp-research/clemcore`. The core framework code lives in `clemcore/` and has (mostly) not been modified, with the exception of transcript styling (clemcore/resources/ — CSS files). All project-specific additions are in `notebooks/`, `archive/`, `final_results/`, and `reports_and_decks/`.

---

## Background: What is clemcore?

[clemcore](https://github.com/clp-research/clemcore) is a framework for testing chat-optimized language models by engaging them as players in structured games games (world-famous (wordle, taboo) as well as manually constructed (privateshared)), both text-based and image-based. A game master exchanges messages with a model, and the model's ability to follow instructions, maintain context, and produce correctly formatted responses is measured.

The games themselves come from a separate repository: clembench. To reproduce my experiments exactly, you will need a set of game instances from my [clembench fork](https://github.com/mariamaninna/clembench).

---

## What this project adds

The core question explored here is: **can we improve a model's game performance on agentic tasks by wrapping it in a smarter agent?**

Five agent architectures are implemented and compared. All agents are defined in `notebooks/agents.ipynb` and can be dropped into any experiment notebook with a single `%run` call.

### Agent architectures

| Agent | Strategy |
|---|---|
| `BaselineAgentPlayer` | Calls the model directly on the interaction history. No modifications. |
| `BasePlanningAgent` | Adds a planning step: asks the model to reason about the next move before acting. Extracts the final action after an `ACT:` separator. |
| `FormatCheckingAgent` | Retries up to 3 times if the model's response is missing required format tags. Tags are read from `game_tags.json`. |
| `FormatCheckingAgent_LLM_judge` | Same as above, but uses a separate LLM to extract the required tags from the game prompt automatically. |
| `UpfrontStrategyAgent` | Injects a short game-specific strategy hint at the start of every turn. Strategies are defined in `game_strategies.json`. |
| `ReflexionAgent` | After each episode, uses an LLM to reflect on what happened. Reflections are stored and injected into future episodes as memory. |

### LangChain-based agents

The targeted modifications above address specific weaknesses through lightweight, interpretable changes. As the opposite end of the architectural spectrum, it is of interest to offload agent behaviour to an established, general-purpose framework — asking whether off-the-shelf agent infrastructure can match or outperform hand-crafted interventions.

For this purpose, we used [LangChain](https://www.langchain.com/), a Python library for building custom agents. LangChain agents are built on the ReAct paradigm (Yao et al., 2022): at each step, the model receives the full conversation history and decides either to call a tool or produce a final response. If a tool is called, its result is appended to the conversation and the model is invoked again; this repeats until a plain-text response is returned. State across turns is managed by a checkpointer, which stores the full message history keyed by a thread id, giving the agent persistent within-episode memory without explicit context management.

There are parallels with clemcore (e.g., the `ClemAgent` history vs. the LangChain checkpointer). However, a LangChain agent without tools is not a meaningful point of comparison: when no tools are defined, the Reason–Act loop degenerates to a single step and the agent behaves identically to a plain chat model. The framework adds no functional capability beyond the baseline.

Three tool configurations were evaluated:
- Agent with a tag extraction tool (simple LM) and a short-term memory component
- Agent with 4 abstract tools for rule following and game observations and a short-term memory component
- Agent with 2 abstract tools for strategy planning and a short-term memory component

LangChain experiments are in `notebooks/langchain_agents.ipynb` and `notebooks/langchain_constructor_joined.ipynb`.

---

## Repository structure

```
notebooks/
    agents.ipynb                   # All agent class definitions — start here
    clem_agent_baseline.ipynb      # Baseline experiment
    clem_agent_experiments.ipynb   # Main experiment notebook
    langchain_agents.ipynb         # LangChain-based agent exploration
    langchain_constructor_joined.ipynb
    game_tags.json                 # Required format tags per game (used by FormatCheckingAgent)
    game_strategies.json           # Strategy hints per game (used by UpfrontStrategyAgent)

archive/
    early_experiments/             # Early notebook drafts
    game_results/                  # Raw interaction logs from runs

final_results/
    results.csv                    # Aggregated results across agents and games
    results.html                   # Results as rendered table

reports_and_decks/                 # Project report and presentation slides

clemcore/                          # Upstream framework code 
```

---

## Installation

Some clemcore elements were modified to handle new architectural add-ons. Please install the version from this repo:

```bash
git clone https://github.com/mariamaninna/clemcore.git
cd clemcore
pip install -e .
```

Then clone the clembench games from my fork (required to reproduce the exact game instances used in these experiments):

```bash
git clone https://github.com/mariamaninna/clembench
```

You can also use the default game instances:

```bash
git clone https://github.com/clp-research/clembench
```

Add your API keys to `key.json` (copy from the provided template).

Install the playpen agent layer (required to run the notebooks):

```bash
pip install playpen
```

---

## Running experiments

Open any notebook in `notebooks/`. Each experiment notebook expects two variables to be set before running:

```python
MODEL = "gpt-4o-mini"   # any model name from model_registry.json
GAME  = "wordle"        # any game available in your clembench folder
```

Then load all agent definitions:

```python
%run ./agents.ipynb
```

See `clem_agent_experiments.ipynb` for a full working example.


---

## Upstream framework

Full documentation for clemcore (CLI, backends, model registry, adding games) is in the original repository: https://github.com/clp-research/clemcore
