# pytest-rag-llm-evaluator

This repository provides a Pytest-based framework designed to evaluate Retrieval-Augmented Generation (RAG) based Large Language Models (LLMs). This tool helps developers and researchers validate the performance, accuracy, and reliability of custom LLM architectures using structured tests and industry-standard metrics.

## 💡Overview

This framework enables end-to-end evaluation of [RAG](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) pipelines, covering:
* 	Retrieval Modules
* 	Augmentation Modules
* 	Generation Modules

It integrates with the RAGAS library, OpenAI, and Langchain to deliver actionable insights on LLM performance.

## ⚙️ Tech stack
1. [Python](https://www.python.org/downloads/)
2. [Pytest](https://docs.pytest.org/en/stable/)
3. [RAGAS](https://docs.ragas.io/) Library
4. [OpenAI API](https://platform.openai.com/)
5. [Langchain](https://python.langchain.com/docs/introduction/)

## 📊 Evaluation Metrics
1. ✅ [Context Precision](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/)
2. ✅ [Context Recall](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/)
3. ✅ [Context Entity Recall](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_entities_recall/#context-entities-recall)
4. ✅ [Faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/)
5. ✅ [Factual Correctness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/factual_correctness/)
6. ✅ [Response Relevancy](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/)
7. ✅ [Topic Adherence](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/agents/#topic_adherence)
8. ✅ [Rubrics Score](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/general_purpose/#rubrics-based-scoring)
9. ✅ [Noise Sensitivity](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/noise_sensitivity/)
10. ✅ [Answer Accuracy](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/nvidia_metrics/#answer-accuracy)
11. ✅ [Context Relevance](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/nvidia_metrics/#context-relevance)
12. ✅ [Semantic Similarity](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/semantic_similarity/)
13. ✅ [Response Groundedness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/nvidia_metrics/#response-groundedness)
14. ✅ [Context Relevance](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/nvidia_metrics/#context-relevance)

## 🔍 Features

1️⃣ Multi-Stage Testing Scope

Test your RAG system at each stage — retrieval, augmentation, and generation — to identify weaknesses early.

3️⃣ Multiple Metrics Evaluation

Leverage EvalDataSet to calculate and benchmark LLM responses against ground truth data using multiple metrics.

4️⃣ Multi-Conversational Scenarios

Simulate real-world multi-turn conversations to test your LLM’s consistency and contextual understanding.

5️⃣ Synthetic Test Data

Generate synthetic question-answer pairs to stress-test your LLM’s capabilities.

6️⃣ Test Optimization

All tests are designed with pytest standards, making it easy to extend, automate, and integrate into CI/CD pipelines.

## 📂 Repository Structure
````plaintext
pytest-rag-llm-evaluator/
├── tests/                # Pytest test cases
├── data/                 # Synthetic & real datasets
├── utils/                # Utility functions for data generation and processing
├── configs/              # Config files for different LLM setups
├── README.md             # Project documentation
└── requirements.txt      # Dependencies
````

