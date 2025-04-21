import pytest

from ragas import SingleTurnSample
from ragas.metrics import ResponseGroundedness

from utils.LLMUtils import get_response_from_llm
from utils.TestDataUtils import load_data_sets


# Terminology:
# Response Groundedness: The degree to which the response is grounded in the retrieved context.
# response: generated answer by the model
# retrieved_context: context retrieved from a knowledge base or database. Eg: Top K relevant documents
@pytest.mark.parametrize("generate_data_feed",
                         load_data_sets("ResponseGroundednessDataFeed.json"),
                         indirect=True)
@pytest.mark.asyncio
async def test_response_groundedness(llm_wrapper, generate_data_feed):
    # Initialize the ResponseGroundedness metric
    response_groundedness = ResponseGroundedness(llm_wrapper)
    # Metric to calculate the response groundedness score
    response_groundedness_score = await response_groundedness.single_turn_ascore(generate_data_feed)
    print("Response Groundedness Score: ", response_groundedness_score)
    assert response_groundedness_score >= 2


@pytest.fixture(request)
def generate_data_feed(request):
    test_data = request.param
    responseFromRAGLLM = get_response_from_llm(test_data)
    # Extract the response and retrieved context
    retrieved_contexts = [doc["page_content"] for doc in responseFromRAGLLM["retrieved_docs"]]

    singleTurnSampleData = SingleTurnSample(
        response=responseFromRAGLLM["answer"],
        retrieved_contexts=retrieved_contexts
    )
    return singleTurnSampleData
