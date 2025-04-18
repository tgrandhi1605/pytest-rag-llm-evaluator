import pytest
from ragas import SingleTurnSample, MultiTurnSample
from ragas.messages import HumanMessage, AIMessage
from ragas.metrics import FactualCorrectness, TopicAdherenceScore

from utils.LLMUtils import get_response_from_llm
from utils.TestDataUtils import load_data_sets


# Terminology:
# Topic Adherence: The degree to which a language model's generated response adheres to the topic of the user's input.
# user_input: question or query by the user
# response: generated answer by the model
@pytest.mark.asyncio
async def test_topic_adherence(llm_wrapper, generate_data_feed):
    factual_correctness = TopicAdherenceScore(llm=llm_wrapper)
    score = await factual_correctness.multi_turn_ascore(generate_data_feed)
    print("Factual Correctness Score: ", score)
    assert score >= 0.8


@pytest.fixture
def generate_data_feed():
    topic_conversation = [
        HumanMessage(content="How many articles are there in selenium webdriver python course"),
        AIMessage(content="There are 23 articles in the course."),
        HumanMessage(content="How many downloadable courses are there in the course?"),
        AIMessage(content="There are 9 downloadable resources in the courses."),
    ]

    reference_topics = [
        """ 
        The AI should:
        1. Understand the user's input and provide relevant information about the course.
        2. Provide accurate information about the number of articles and downloadable resources in the course.
        3. Maintain a conversational tone and engage with the user.
        """
    ]

    multiTurnSample = MultiTurnSample(
        user_input=topic_conversation,
        reference_topics=reference_topics,
    )
    return multiTurnSample
