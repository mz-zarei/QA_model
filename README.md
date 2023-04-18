# Q/A Over Documents
This module provides functions to prepare and process text documents, perform document search, prepare language models, and generate answers based on the documents and a given query.

#Installation
This code requires Python 3.7 or later versions and the following dependencies:

- torch
- transformers
- langchain
- huggingface_hub
- sentence_transformers
- chromadb
- accelerate
- bitsandbytes

To install the dependencies, you can use the following command:

`!pip -q install langchain huggingface_hub transformers sentence_transformers chromadb accelerate bitsandbytes`


# Usage
This module provides a command-line interface to retrieve answers to queries over text documents. You can use the following command to run the code:


`!python run_qa.py --query-path query.txt --text-path texts.txt --model-id google/flan-t5-xl --k 2 --chunk-size 64 --chunk-overlap 0`

## Arguments
The following arguments can be used to customize the behavior of the code:

- --query-path: Path to the file containing the query text. Default is './query.txt'.
- --text-path: Path to the file containing the input text documents. Default is './texts.txt'.
- --model-path: Path to the local model weights. If not provided, the default HuggingFace model will be used.
- --model-id: HuggingFace model ID. Default is 'google/flan-t5-xl'.
- --k: Number of top related documents to be considered. Default is 2.
- --chunk-size: Size of each text split. Default is 64.
- --chunk-overlap: Overlap between text splits. Default is 0.

## Functions
This module provides the following functions:

- prepare_text(path: str='texts.txt', chunk_size:int=100, chunk_overlap:int=0) -> List[str]: Load and split the text file into chunks.
find_docs(query:str, path: str='texts.txt', k:int=2, chunk_size:int=64, chunk_overlap:int=0) -> List[Dict]: Find relevant documents for a given query.
- prepare_llm(model_id:str='google/flan-t5-xl') -> HuggingFacePipeline: Load the language model.
- get_answer(query:str, path: str='texts.txt', k:int=2, model_id:str='google/flan-t5-xl', chunk_size:int=64, chunk_overlap:int=0) -> str: Retrieves an answer for the given query using the specified model and settings.

# Acknowledgments
This code is based on the LangChain framework developed by Rasa, a conversational AI company.




Regenerate response
