import pytest
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import FactualCorrectness, ResponseRelevancy, LLMContextPrecisionWithoutReference, \
    LLMContextRecall, Faithfulness

from conftest import llm_wrapper
from utils.LLMUtils import get_response_from_llm
from utils.TestDataUtils import load_data_sets


# This test is to evaluate the performance of the LLM using multiple metrics with in a single test case.
# Metrics:
# 1. Factual Correctness
# 2. Response Relevancy
# 3. Context Precision
# 4. Context Recall
# 5. Faithfulness

@pytest.mark.asyncio
@pytest.mark.parametrize("generate_data_feed",
                         load_data_sets("ResponseRelevancyDataFeed.json"),
                         indirect=True)
async def test_response_relevance(llm_wrapper, generate_data_feed):
    metrics_to_evaluate = [FactualCorrectness(llm=llm_wrapper),
                           ResponseRelevancy(llm=llm_wrapper),
                           LLMContextPrecisionWithoutReference(llm=llm_wrapper),
                           LLMContextRecall(llm=llm_wrapper),
                           Faithfulness(llm=llm_wrapper)]

    data_set = EvaluationDataset([generate_data_feed])

    # metrics_to_evaluate are the list of specific metrics to evaluate
    metrics_scores = evaluate(data_set, metrics_to_evaluate)

    # If we don't pass any metrics, it will evaluate the standard metrics such as answer relevancy, context_precision, context_recall, and faithfulness
    # metrics_scores = evaluate(data_set)

    print("Metrics Scores: ", metrics_scores)

    # Response Relevancy
    print("Response Relevancy Score: ", metrics_scores["answer_relevancy"])
    assert metrics_scores["answer_relevancy"] > 0.95

    # Factual Correctness
    print("Factual Correctness Score: ", metrics_scores["factual_correctness"])
    assert metrics_scores["factual_correctness"] > 0.95

    # Context Precision
    print("Context Precision Score: ", metrics_scores["context_precision"])
    assert metrics_scores["context_precision"] > 0.95

    # Context Recall
    print("Context Recall Score: ", metrics_scores["context_recall"])
    assert metrics_scores["context_recall"] > 0.95

    # Faithfulness
    print("Faithfulness Score: ", metrics_scores["faithfulness"])
    assert metrics_scores["faithfulness"] > 0.95

    # Upload the metrics scores to RAGAS app portal
    metrics_scores.upload()


@pytest.fixture
def generate_data_feed(request):
    test_data = request.param
    responseFromRAGLLM = get_response_from_llm(test_data)
    # Extract the response and retrieved context
    retrieved_contexts = [doc["page_content"] for doc in responseFromRAGLLM["retrieved_docs"]]

    singleTurnSampleData = SingleTurnSample(
        user_input=test_data["question"],
        reference=test_data["reference"], # Ground truth or actual answer for a given question or user input
        retrieved_contexts=retrieved_contexts,
        response=responseFromRAGLLM["answer"],
    )
    return singleTurnSampleData
