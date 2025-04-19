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
3. ✅ [Faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/)
4. ✅ [Factual Correctness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/factual_correctness/)
5. ✅ [Response Relevancy](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/)
6. ✅ [Topic Adherence](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/agents/#topic_adherence)
7. ✅ [Rubrics Score](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/general_purpose/#rubrics-based-scoring)

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
├── metrics/              # Metric definitions and assertions
├── configs/              # Config files for different LLM setups
├── README.md             # Project documentation
└── requirements.txt      # Dependencies`
````

