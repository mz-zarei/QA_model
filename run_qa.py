"""
Language Model-based Question Answering System

This script provides functions to prepare and process text documents, perform document search,
prepare language models, and generate answers based on the documents and a given query.

Author: Mohammad Zarei
Date: 2023-04-17

"""

# imports
import argparse
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)

from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain



# functions
def prepare_text(path: str='texts.txt', 
                 chunk_size:int=100, 
                 chunk_overlap:int=0) -> List[str]:
    """
    Load and split the text file into chunks.

    Args:
        path (str): The path to the text file. Default is 'texts.txt'.
        chunk_size (int): The size of each text chunk. Default is 100.
        chunk_overlap (int): The number of overlapping characters between chunks. Default is 0.

    Returns:
        List[str]: A list of text chunks.
    """

    with open(path, encoding='utf-8') as f:
        res = f.read()

    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(res)
    return texts

def find_docs(query:str, 
              path: str='texts.txt', 
              k:int=2, 
              chunk_size:int=64, 
              chunk_overlap:int=0) -> List[Dict]:
    """
    Find relevant documents for a given query.

    Args:
        query (str): The query to search for.
        path (str): The path to the text file. Default is 'texts.txt'.
        k (int): The number of top search results to return. Default is 2.
        chunk_size (int): The size of each text chunk. Default is 100.
        chunk_overlap (int): The number of overlapping characters between chunks. Default is 0.

    Returns:
        List[Dict]: A list of documents with their similarity scores and metadata.
    """
    texts = prepare_text(path, chunk_size, chunk_overlap)
    embeddings = HuggingFaceEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()
    docs = docsearch.vectorstore.similarity_search(query, k=k)
    return docs

def prepare_llm(model_id:str='google/flan-t5-xl') -> HuggingFacePipeline:
    """
    Load the Language model

    Args:
        model_id (str): The model id to be used. Default is 'google/flan-t5-xl'.

    Returns:
        HuggingFacePipeline: The language model pipeline object.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH if MODEL_PATH else model_id, local_files_only=bool(MODEL_PATH)
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH if MODEL_PATH else model_id,
        local_files_only=bool(MODEL_PATH),
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map='auto',
    )

    pipe = pipeline(
                    "text2text-generation",
                    model=model, 
                    tokenizer=tokenizer, 
                    # max_length=1000,
                    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def get_answer(query:str, 
               path: str='texts.txt',
               k:int=2, 
               model_id:str='google/flan-t5-xl',
               chunk_size:int=64, 
               chunk_overlap:int=0) -> str:
    """
    Retrieves an answer for the given query using the specified model and settings.

    Args:
        query (str): The question to be answered.
        path (str, optional): Path to the input text file. Defaults to 'texts.txt'.
        k (int, optional): The number of top documents to be considered. Defaults to 2.
        model_id (str, optional): The ID of the pretrained model to use. Defaults to 'google/flan-t5-xl'.
        chunk_size (int, optional): The size of the text chunks. Defaults to 64.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 0.

    Returns:
        str: The answer to the given query.
    """

    template = """
    The full sentence asnwer to question:
    {question}
    
    based on following summary:
    {summaries}

    is the following:
    """

    docs = find_docs(query, path, k, chunk_size, chunk_overlap)
    PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

    chain = load_qa_with_sources_chain(llm=prepare_llm(model_id),
                                    chain_type="stuff", 
                                    prompt=PROMPT)
    return chain({"input_documents": docs, "question": query}, return_only_outputs=True)



parser = argparse.ArgumentParser(description='Q/A over docs')
parser.add_argument('--query-path', type=str, default='./query.txt',
                    help='path to query text file')
parser.add_argument('--text-path', type=str, default='./texts.txt',
                    help='path to text file')
parser.add_argument('--model-path', type=str, default= None,
                    help='path to local model weights')
parser.add_argument('--model-id', type=str, default='google/flan-t5-xl',
                    help='HuggingFace model ID')
parser.add_argument('--k', type=int, default=2,
                    help='number of related docs')
parser.add_argument('--chunk-size', type=int, default=64,
                    help='size of each text split')
parser.add_argument('--chunk-overlap', type=int, default=0,
                    help='text splits overlap')        
args = parser.parse_args()

# settings
QUERY_PATH = args.query_path
TEXT_PATH = args.text_path
MODEL_PATH = args.model_path
MODEL_ID = args.model_id
K = args.k
CHUNK_SIZE = args.chunk_size
CHUNK_OVERLAP = args.chunk_overlap


def main():
    with open(QUERY_PATH, 'r') as file:
        query = file.read().replace('\n', ' ')
    
    answer = get_answer(query=query, 
                        path=TEXT_PATH, 
                        k=K, 
                        model_id=MODEL_ID,
                        chunk_size=CHUNK_SIZE, 
                        chunk_overlap=CHUNK_OVERLAP)
    print("Answer: ", answer['output_text'])
    
if __name__ == '__main__':
    main()