# pytest-rag-llm-evaluator

This repository provides a Pytest-based framework designed to evaluate Retrieval-Augmented Generation (RAG) based Large Language Models (LLMs). This tool helps developers and researchers validate the performance, accuracy, and reliability of custom LLM architectures using structured tests and industry-standard metrics.

## 💡Overview

This framework enables end-to-end evaluation of RAG pipelines, covering:
* 	Retrieval Modules
* 	Augmentation Modules
* 	Generation Modules

It integrates with the RAGAS library, OpenAI, and Langchain to deliver actionable insights on LLM performance.

## ⚙️ Tech stack
1. Python
2. Pytest
3. RAGAS Library
4. OpenAI API
5. Langchain

## 📊 Evaluation Metrics
1. ✅ Context Precision
2. ✅ Context Recall
3. ✅ Faithfulness
4. ✅ Factual Correctness
5. ✅ Response Relevancy
6. ✅ Topic Adherence
7. ✅ Rubrics Score

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

