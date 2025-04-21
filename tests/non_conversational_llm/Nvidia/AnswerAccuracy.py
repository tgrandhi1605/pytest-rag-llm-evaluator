import pytest

from ragas import SingleTurnSample
from ragas.metrics import AnswerAccuracy

from utils.LLMUtils import get_response_from_llm
from utils.TestDataUtils import load_data_sets


@pytest.mark.parametrize("generate_data_feed",
                         load_data_sets("AnswerAccuracyDataFeed.json"),
                         indirect=True)
@pytest.mark.asyncio
async def test_answer_accuracy(llm_wrapper, generate_data_feed):
    # Initialize the AnswerAccuracy metric
    answer_accuracy = AnswerAccuracy(llm_wrapper)
    # Metric to calculate the answer accuracy score
    answer_accuracy_score = await answer_accuracy.single_turn_ascore(generate_data_feed)
    print("Answer Accuracy Score: ", answer_accuracy_score)
    assert answer_accuracy_score >= 2


@pytest.fixture
def generate_data_feed(request):
    test_data = request.param
    responseFromRAGLLM = get_response_from_llm(test_data)

    singleTurnSampleData = SingleTurnSample(
        user_input=test_data["question"],
        reference=test_data["reference"], # Ground truth or actual answer for a given question or user input
        response=responseFromRAGLLM["answer"]
    )
    return singleTurnSampleData
