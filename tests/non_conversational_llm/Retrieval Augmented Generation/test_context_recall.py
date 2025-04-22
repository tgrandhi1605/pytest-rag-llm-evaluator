import pytest

from ragas import SingleTurnSample
from ragas.metrics import LLMContextRecall

from utils.LLMUtils import get_response_from_llm
from utils.TestDataUtils import load_data_sets


# Terminology:
# LLM: Large Language Model
# Context recall: The degree to which a language model can accurately recall relevant information from the context provided.
# user_input: question or query by the user
# reference: expected answer or ground truth
# retrieved_context: context retrieved from a knowledge base or database. Eg: Top K relevant documents

@pytest.mark.asyncio
@pytest.mark.parametrize("generate_data_feed",
                         load_data_sets("ContextRecallDataFeed.json"),
                         indirect=True)
async def test_context_recall(llm_wrapper, generate_data_feed):
    # Initialize the LLMContextRecall metric
    context_recall = LLMContextRecall(llm=llm_wrapper)
    # Metric to calculate the context recall score
    context_recall_score = await context_recall.single_turn_ascore(generate_data_feed)
    print("Context Recall Score: ", context_recall_score)
    assert context_recall_score >= 0.95


@pytest.fixture
def generate_data_feed(request):
    test_data = request.param
    responseFromRAGLLM = get_response_from_llm(test_data)
    # Extract the response and retrieved context
    retrieved_contexts = [doc["page_content"] for doc in responseFromRAGLLM["retrieved_docs"]]

    singleTurnSampleData = SingleTurnSample(
        user_input=test_data["question"],
        reference=test_data["reference"], # Ground truth or actual answer for a given question or user input
        retrieved_contexts=retrieved_contexts,
    )
    return singleTurnSampleData
