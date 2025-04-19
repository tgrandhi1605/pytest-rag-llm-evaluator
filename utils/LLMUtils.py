from langchain import requests


def get_response_from_llm(test_data):
    return requests.post("Replace with your LLM endpoint",
                         # Replace with your own schema for the LLM endpoint
                         json={
                             "question": test_data["question"],
                             "chat_history": []
                         }).json()
