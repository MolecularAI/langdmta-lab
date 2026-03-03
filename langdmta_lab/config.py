import os

LANGDMTA_ROOT_DIR = os.path.dirname(__file__)
LANGDMTA_GRAPH_DIR = os.path.join(LANGDMTA_ROOT_DIR, "multiagent")
LANGDMTA_MCPS_DIR = os.path.join(LANGDMTA_ROOT_DIR, "mcps")
TEST_QUESTION_DIR = os.path.join(os.path.dirname(LANGDMTA_ROOT_DIR), "tests")
TEST_ASSETS_DIR = os.path.join(TEST_QUESTION_DIR, "assets")
ENV_DIR = os.path.join(os.path.dirname(LANGDMTA_ROOT_DIR), "envs")

AZURE_OPENAI_GPT4o_VERSION = "2025-01-01-preview"

END_TOKEN = "FINISH"
RESPONSE_PREFIX = "Response from"
PARSING_TAG = "<parsing required>"
