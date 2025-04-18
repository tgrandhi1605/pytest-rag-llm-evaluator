from langchain import requests


def get_response_from_llm(test_data):
    return requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                         json={
                             "question": test_data["question"],
                             "chat_history": []
                         }).json()
