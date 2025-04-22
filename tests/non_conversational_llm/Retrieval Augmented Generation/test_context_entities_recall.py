import pytest

from ragas import SingleTurnSample
from ragas.metrics import ContextEntityRecall

from utils.LLMUtils import get_response_from_llm
from utils.TestDataUtils import load_data_sets


# Terminology:
# Context Entity Recall: The degree to which a language model can accurately recall relevant entities from the context provided.
# reference: expected answer or ground truth
# retrieved_context: context retrieved from a knowledge base or database. Eg: Top K relevant documents
@pytest.mark.parametrize("generate_data_feed",
                         load_data_sets("ContextEntityRecallDataFeed.json"),
                         indirect=True)
@pytest.mark.asyncio
async def test_context_entity_recall(llm_wrapper, generate_data_feed):
    # Initialize the ContextEntityRecall metric
    context_entity_recall = ContextEntityRecall(llm_wrapper)
    # Metric to calculate the context entity recall score
    context_entity_recall_score = await context_entity_recall.single_turn_ascore(generate_data_feed)
    print("Context Entity Recall Score: ", context_entity_recall_score)
    assert context_entity_recall_score >= 0.75


@pytest.fixture()
def generate_data_feed(request):
    test_data = request.param
    responseFromRAGLLM = get_response_from_llm(test_data)
    # Extract the response and retrieved context
    retrieved_contexts = [doc["page_content"] for doc in responseFromRAGLLM["retrieved_docs"]]

    singleTurnSampleData = SingleTurnSample(
        reference=test_data["reference"], # Ground truth or actual answer for a given question or user input
        retrieved_contexts=retrieved_contexts
    )
    return singleTurnSampleData
