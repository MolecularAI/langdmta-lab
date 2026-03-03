# langdmta-lab

This repository contains the code for an LLM-based agentic system for drug discovery. The agent has
a multi-agent architecture, with a supervisor agent delegating the work to four sub-agents:

- Design - an agent that can generate novel compounds and score compounds
- Synthesis - an agent that can assist in the planning of the chemical synthesis of compounds
- Analyzer - an agent that can analyze an output file in CSV format in order to compute statistics or make plots
- Utility - an agent that assists in miscellaneous tasks such as converting common names of compounds to chemical structures


## Prerequisites
Before you begin, ensure you have met the following requirements:

Only Linux platforms are supported.

You have installed a conda distribution with python 3.12

## Setup

Setting up the conda environment

    conda create -n langlab python=3.12 uv -y
    conda activate langlab
    uv pip install -e .

Setting up the MCP inspector

    conda activate langlab
    conda install -c conda-forge nodejs

Setting up the environment for the Design MCP server

    cd envs/design
    bash install.sh

Setting up the environment for the Synthesis MCP server

    cd envs/synthesis
    bash install.sh

*Note! For now it is better to run the commands in install.sh manually because conda activate does not work well*

In addition to this, we use [LangFuse](https://langfuse.com/) to track agent executions. You need to set up a LangFuse
account and project to enable use of the evaluation pipeline.

## MCP inspector

In the terminal, type

    conda activate langlab
    npx @modelcontextprotocol/inspector

and it should automatically open up the MCP inspector in your browser

## Secrets

Secrets need to be put in `.env` in the repo base folder

    export OPENAI_API_KEY=""

    export LANGFUSE_USER_ID=""
    export LANGFUSE_SECRET_KEY=""
    export LANGFUSE_PUBLIC_KEY=""
    export LANGFUSE_HOST=""

    export DESIGN_MCP_PORT=""
    export SYNTHESIS_MCP_PORT=""
    export UTILITY_MCP_PORT=""

## Running MCP servers

    cd langdmta_lab/mcps
    USE_MOCK=true bash init_mcps.sh

to use with mocked data, or

    bash init_mcps.sh

to use the tools to generate output

## Tests and evaluation


Run the test questions with a useful session id for Langfuse

     python tests/test_questions.py --session-id mysession --num-trials 2

Extract from Langfuse and summarise it

    python evaluation/langfuse_summary.py --session-ids mysession_trial_1,mysession_trial_2

**Note:** The MCP servers will be started automatically

## Contributors

* [@jkinga](https://www.github.com/jkinga)
* [@emmaryd](https://www.github.com/emmaryd)
* [@lakshidaa](https://github.com/Lakshidaa)
* [@helenlai](https://github.com/helenlai)
* [@SGenheden](https://www.github.com/SGenheden)

## License

The software is licensed under the Apache 2.0 license (see LICENSE file), and is free and provided as-is.

## References

He, J., Lai, H., Saigiridharan, L., Ghiandoni, G.M., Jenei, K., Gokalp, U., Nukovic, A., Engkvist, O., Janet, J.P., & Genheden, S. (2026). Democratising real-world drug discovery through agentic AI. Drug discovery today, 104605 . [10.1016/j.drudis.2026.104605](https://doi.org/10.1016/j.drudis.2026.104605)
