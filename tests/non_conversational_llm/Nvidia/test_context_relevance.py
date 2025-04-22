import pytest
from ragas import SingleTurnSample
from ragas.metrics import ContextRelevance

from utils.LLMUtils import get_response_from_llm
from utils.TestDataUtils import load_data_sets


# Tolerance:
# Context Relevance: The degree to which the retrieved context is relevant to the user's input.
# user_input: question or query by the user
# retrieved_context: context retrieved from a knowledge base or database. Eg: Top K relevant documents
@pytest.mark.parametrize("generate_data_feed",
                         load_data_sets("ContextRelevanceDataFeed.json"),
                         indirect=True)
@pytest.mark.asyncio
async def test_context_relevance(llm_wrapper, generate_data_feed):
    # Initialize the ContextRelevance metric
    context_relevance = ContextRelevance(llm_wrapper)
    # Metric to calculate the context relevance score
    context_relevance_score = await context_relevance.single_turn_ascore(generate_data_feed)
    print("Context Relevance Score: ", context_relevance_score)
    assert context_relevance_score >= 2

@pytest.fixture
def generate_data_feed(request):
    test_data = request.param
    responseFromRAGLLM = get_response_from_llm(test_data)
    # Extract the response and retrieved context
    retrieved_contexts = [doc["page_content"] for doc in responseFromRAGLLM["retrieved_docs"]]

    singleTurnSampleData = SingleTurnSample(
        user_input=test_data["question"],
        retrieved_contexts=retrieved_contexts
    )
    return singleTurnSampleData

