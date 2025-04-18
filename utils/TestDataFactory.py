from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
import nltk


def data_generator():
    nltk.data.path.append("/Users/tgrandhi/nltk_data") # Add your NLTK data path here where tokenizers & taggers are placed in your local
    base_llm = ChatOpenAI(model="gpt-4", temperature=0)
    ragas_llm = LangchainLLMWrapper(base_llm)
    embeddings = OpenAIEmbeddings()
    document_loader = DirectoryLoader(
        path="/Users/tgrandhi/Downloads/LLM Evaluation_Resources/fs11",
        glob="**/*.docx",
        show_progress=True,
    )
    documents = document_loader.load()
    generate_embeddings = LangchainEmbeddingsWrapper(embeddings)
    generator = TestsetGenerator(llm=ragas_llm, embedding_model=generate_embeddings)
    dataset = generator.generate_with_langchain_docs(documents, testset_size=10)
    print("Generated Dataset: ", dataset)
    dataset.upload()
