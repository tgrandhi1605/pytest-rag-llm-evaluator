import pytest
from ragas import SingleTurnSample
from ragas.metrics import RubricsScore

from utils.LLMUtils import get_response_from_llm


@pytest.mark.asyncio
async def test_rubrics_score(llm_wrapper, generate_data_feed):
    # source: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/general_purpose/#rubrics-based-criteria-scoring
    rubrics = {
        "score1_description": "The response is entirely incorrect and fails to address any aspect of the reference.",
        "score2_description": "The response contains partial accuracy but includes major errors or significant omissions that affect its relevance to the reference.",
        "score3_description": "The response is mostly accurate but lacks clarity, thoroughness, or minor details needed to fully address the reference.",
        "score4_description": "The response is accurate and clear, with only minor omissions or slight inaccuracies in addressing the reference.",
        "score5_description": "The response is completely accurate, clear, and thoroughly addresses the reference without any errors or omissions."
    }

    # Initialize the RubricScore metric
    rubrics_score = RubricsScore(rubrics=rubrics, llm=llm_wrapper)

    # Fetch the score
    score = await rubrics_score.single_turn_ascore(generate_data_feed)
    print("Rubric Score: ", score)
    assert score > 0.7


@pytest.fixture
def generate_data_feed(request):
    test_data = request.param
    responseFromRAGLLM = get_response_from_llm(test_data)
    # Extract the response and retrieved context
    retrieved_contexts = [doc["page_content"] for doc in responseFromRAGLLM["retrieved_docs"]]

    singleTurnSampleData = SingleTurnSample(
        user_input="Where is Eiffel tower located?",
        response="Eiffel Tower is located in France.",
        reference="Eiffel Tower is located in Paris, France.",
    )
    return singleTurnSampleData
