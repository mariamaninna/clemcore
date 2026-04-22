Experiments conducted so far:

0) Baseline

A ClemAgent playing clemgames

1) LangChain-based

- Agent with a tag extraction tool (simple LM) and a short-term memory component
- Agent with 4 abstract tools for rule following and game observations and a short-term memory component
- Agent with 2 abstract tools for strategy planning and a short-term memory component


2) Base ClemAgent with add-ons

- Agent with an additional planning loop: queries the model for an ACT: response, then the answer is passed to the game environment.


Installation:

Some clemcore elements were modified to handle new architectural add-ons. Please install the version from this repo:

git clone https://github.com/mariamaninna/clemcore.git
cd clemcore
pip install -e .