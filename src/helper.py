#load the documents

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_classic.schema import Document

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents



def filter_to_minimal_doc(docs:list[Document])->list[Document]:
    """
    given a list of document objects , return a new list of document objects
    conataing only 'source' in metadata and the orginaln page_content.
    """
    minimal_doc : list[Document]=[]
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_doc.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_doc    


#split data into text chunks
def text_split(extracted_data):
    text_spillter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunk=text_spillter.split_documents(extracted_data)
    return text_chunk


# download the embedding model from huggingface
 
def download_embedding():
    """downlaod and return hgging face model"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
    )
    return embeddings
#embeddings = download_embedding()