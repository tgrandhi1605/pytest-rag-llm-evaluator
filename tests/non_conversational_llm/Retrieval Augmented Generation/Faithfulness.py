import pytest
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness

from utils.LLMUtils import get_response_from_llm
from utils.TestDataUtils import load_data_sets


# Terminology:
# Faithfulness: The degree to which a language model's generated response is consistent with the information provided in the context.
# user_input: question or query by the user
# response: generated answer by the model
# retrieved_context: context retrieved from a knowledge base or database. Eg: Top K relevant documents
@pytest.mark.parametrize("generate_data_feed",
                         load_data_sets("FailthfulnessDataFeed.json"),
                         indirect=True)
@pytest.mark.asyncio
async def test_faithfulness(llm_wrapper, generate_data_feed):
    faithfulness = Faithfulness(llm=llm_wrapper)
    faithfulness_score = await faithfulness.single_turn_ascore(generate_data_feed)
    print("Faithfulness Score: ", faithfulness_score)
    assert faithfulness_score >= 0.95


@pytest.fixture
def generate_data_feed(request):
    test_data = request.param
    responseFromRAGLLM = get_response_from_llm(test_data)
    # Extract the response and retrieved context
    retrieved_contexts = [doc["page_content"] for doc in responseFromRAGLLM["retrieved_docs"]]

    singleTurnSampleData = SingleTurnSample(
        user_input=test_data["question"],
        response=responseFromRAGLLM["answer"],
        retrieved_contexts=retrieved_contexts,
    )
    return singleTurnSampleData
