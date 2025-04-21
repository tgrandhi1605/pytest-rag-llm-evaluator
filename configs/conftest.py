import os

import pytest
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

os.environ["OPENAI_API_KEY"] = "Enter your OpenAI API key here"  # Replace with your OpenAI API key

os.environ["RAGAS_APP_TOKEN"] = "Enter your RAGAS app token here"  # Replace with your RAGAS app token


@pytest.fixture
def llm_wrapper():
    # Initialize the base LLM & model of your choice
    base_llm = ChatOpenAI(model="gpt-4", temperature=0)
    # Initialize the Langchain LLM Wrapper consuming base OpenAi LLM
    ragas_llm = LangchainLLMWrapper(base_llm)
    return ragas_llm
