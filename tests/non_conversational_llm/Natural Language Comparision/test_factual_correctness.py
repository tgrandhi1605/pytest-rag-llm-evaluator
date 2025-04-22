import pytest
from ragas import SingleTurnSample
from ragas.metrics import FactualCorrectness

from utils.LLMUtils import get_response_from_llm
from utils.TestDataUtils import load_data_sets


# Terminology:
# Factual Correctness: The degree to which a language model's generated response is factually accurate and consistent with the information provided in the context.
# user_input: question or query by the user
# response: generated answer by the model
@pytest.mark.parametrize("generate_data_feed",
                         load_data_sets("FactualCorrectnessDataFeed.json"),
                         indirect=True)
@pytest.mark.asyncio
async def test_factual_correctness(llm_wrapper, generate_data_feed):
    factual_correctness = FactualCorrectness(llm=llm_wrapper)
    factual_correctness_score = await factual_correctness.single_turn_ascore(generate_data_feed)
    print("Factual Correctness Score: ", factual_correctness_score)
    assert factual_correctness_score >= 0.95


@pytest.fixture
def generate_data_feed(request):
    test_data = request.param
    responseFromRAGLLM = get_response_from_llm(test_data)

    singleTurnSampleData = SingleTurnSample(
        user_input=test_data["question"],
        response=responseFromRAGLLM["answer"],
    )
    return singleTurnSampleData
