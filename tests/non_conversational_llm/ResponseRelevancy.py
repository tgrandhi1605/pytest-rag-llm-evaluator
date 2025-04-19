import pytest
from ragas import SingleTurnSample
from ragas.metrics import AnswerRelevancy

from conftest import llm_wrapper
from utils.LLMUtils import get_response_from_llm
from utils.TestDataUtils import load_data_sets


# Terminology:
# Response Relevancy: The degree to which the response is relevant to the user's input.
# AKA: Answer Relevancy
# user_input: question or query by the user
# response: generated answer by the model
@pytest.mark.parametrize("generate_data_feed",
                         load_data_sets("ResponseRelevancyDataFeed.json"),
                         indirect=True)
@pytest.mark.asyncio
async def test_response_relevancy(llm_wrapper, generate_data_feed):
    response_relevancy = AnswerRelevancy(llm=llm_wrapper())
    response_relevancy_score = await response_relevancy.single_turn_ascore(generate_data_feed)
    print("Response Relevancy Score: ", response_relevancy_score)
    assert response_relevancy_score >= 0.95


@pytest.fixture
def generate_data_feed(request):
    test_data = request.param
    responseFromRAGLLM = get_response_from_llm(test_data)

    singleTurnSampleData = SingleTurnSample(
        user_input=test_data["question"],
        response=responseFromRAGLLM["answer"],
    )
    return singleTurnSampleData
