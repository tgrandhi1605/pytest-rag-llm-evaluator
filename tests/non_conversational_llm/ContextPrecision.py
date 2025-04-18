import pytest

from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference

from utils.LLMUtils import get_response_from_llm
from utils.TestDataUtils import load_data_sets


# Terminology:
# Context Precision: The degree to which the context provided to a language model is relevant and accurate.
# user_input: question or query by the user
# response: generated answer by the model
# retrieved_context: context retrieved from a knowledge base or database. Eg: Top K relevant documents

@pytest.mark.asyncio
@pytest.mark.parametrize("generate_data_feed",
                         load_data_sets("ContextPrecisionDataFeed.json"),
                         indirect=True)
async def test_context_precision(llm_wrapper, generate_data_feed):
    # Initialize the LLMContextPrecisionWithoutReference metric
    context_precision_without_reference = LLMContextPrecisionWithoutReference(llm=llm_wrapper)

    # Fetch the score
    contextPrecisionScore = await context_precision_without_reference.single_turn_ascore(generate_data_feed)
    print("Context Precision Score: ", contextPrecisionScore)
    assert contextPrecisionScore >= 0.95


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
