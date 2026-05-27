import re


qa = None

def initialize_qa():
    global qa
    from time import time
    import torch
    import transformers
    from torch import cuda
    from transformers import AutoTokenizer
    from langchain_community.llms import HuggingFacePipeline
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain_community.vectorstores import Chroma
    #Local LLM address
    model_id = '/public/home/duk_wpaul_srs1/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct'

    #Set the device
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    print(device)


    #Time the loading of model
    time_start = time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    #Load AutoModelForCausalLM
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map='auto',
    )

    time_end = time()
    print(f"Prepare model, tokenizer: {round(time_end-time_start, 3)} sec.")


    #Time the construction of pipeline
    time_start = time()

    #construct the pipeline
    query_pipeline = transformers.pipeline(
            "text-generation",#text generation pipeline
            model=model,#load the model
            tokenizer=tokenizer,#load the token
            torch_dtype=torch.float32,
            max_length=512,#maximum length is set to 512
            device_map="auto",
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id,
    )

    time_end = time()
    print(f"Prepare pipeline: {round(time_end-time_start, 3)} sec.")


    #Change to Huggingface pipeline
    llm = HuggingFacePipeline(pipeline=query_pipeline)
    print("Huggingface pipeline construction is done!")


    from sentence_transformers import SentenceTransformer

    #Load the bulletin
    loader = PyPDFLoader("/public/home/duk_wpaul_srs1/ug_bulletin_2023-24.pdf")
    documents = loader.load()

    #Instantiate the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=256)
    #Split all the data
    all_splits = text_splitter.split_documents(documents)


    #Load the sentence transformer
    model_name = SentenceTransformer('/public/home/duk_wpaul_srs1/.cache/modelscope/hub/AI-ModelScope/all-mpnet-base-v2')
    model_kwargs = {"device": "cuda"}

    #Load the embedding model
    local_model_path = '/public/home/duk_wpaul_srs1/.cache/modelscope/hub/AI-ModelScope/all-mpnet-base-v2'
    print(f"Use local model: {local_model_path}\n")
    embeddings = HuggingFaceEmbeddings(model_name=local_model_path, model_kwargs=model_kwargs)

    #Construct the vector database
    vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

    #Integrate the retriever into the vector database
    #Compare the query vector with the vectors in the database and find the most relevant pieces
    retriever = vectordb.as_retriever(search_kwargs={"k": 1})

    #Construct Q&A system
    qa = RetrievalQA.from_chain_type(
        llm=llm, #Load the instantiated model, in terms of Huggingface pipeline
        #The advantage of using "stuff" chain is to comprehensively using the information from multiple documents, in order to improve the system's understanding of the question and the accuracy of the answer. On the other hand, it might also lead to redundant information and noise, which should be adjusted in practical applications.
        chain_type="stuff", #"chain_type" parameter is used to assign the type of chain that organizes the retrieved documents. Using the "stuff" chain, it will connect the retrieved documents as a single input to the LLM.
        retriever=retriever, #Load the retrieval information
        verbose=True,#When set to be True, ask RetrievalQA instance to print out the information about its operations
    )


    return qa

# By running the function `dataset_dict()`, we generate the fundamental dataset structure
# in Python dictionary format.
# The dictionary contains `question`, `ground_truths`, `contexts`, and `answer` as keys.
# By the end of the function, we will have a dictionary with `question` and
# `ground_truths` filled.
def _extract_helpful_answer(raw_answer: str) -> str:
    """Extract the model answer following the `Helpful Answer:` marker."""
    match = re.search(r"Helpful Answer: (.*?)(\n\n|\Z)", raw_answer, re.S)
    if not match:
        raise ValueError(f"Could not find the helpful answer in the output: {raw_answer}")
    return match.group(1).strip()


def dataset_dict(question_answer_file):
    if qa is None:
        raise RuntimeError("qa is not initialized. Call initialize_qa() before dataset generation.")
    data_raw = {
        "question": [],
        "ground_truths": [],
        "contexts": [],
        "answer": [],
    }

    # Fill in `question` and `ground_truths` from the source file.
    with open(question_answer_file, "r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line == "":
                continue  # Skip empty lines.
            if stripped_line.startswith("#"):
                new_question = stripped_line[1:].strip()
                data_raw["question"].append(new_question)
            else:
                data_raw["ground_truths"].append(stripped_line)

    # Fill in `answer` and `contexts`.
    for query in data_raw["question"]:
        new_answer = qa.run(query)
        data_raw["answer"].append(_extract_helpful_answer(new_answer))

        retrieved_docs = qa.retriever.get_relevant_documents(query)
        query_contexts = [doc.page_content.split("\n") for doc in retrieved_docs]
        data_raw["contexts"].append(query_contexts)

    return data_raw

# Store the dictionary in a json file
import json
if __name__ == "__main__":
    initialize_qa()
    dataset = dataset_dict("/public/share/duk_wpaul/github_srs2024_wpaul/Question Set/question.txt")
    with open("answer_1.json", "w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)
