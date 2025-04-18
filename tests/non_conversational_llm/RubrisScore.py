import pytest
from ragas import SingleTurnSample
from ragas.metrics import RubricsScore

from utils.LLMUtils import get_response_from_llm


@pytest.mark.asyncio
async def test_rubrics_score(llm_wrapper, generate_data_feed):

    rubrics = {
        "score_1_description": "The response is completely irrelevant to the question.",
        "score_2_description": "The response is somewhat relevant but lacks depth.",
        "score_3_description": "The response is relevant and provides some depth.",
        "score_4_description": "The response is relevant and provides good depth.",
        "score_5_description": "The response is highly relevant and provides excellent depth.",
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