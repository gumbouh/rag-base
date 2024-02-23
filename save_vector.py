import os
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from chromadb.utils import embedding_functions
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings 
import re

def process_data():
    df = pd.read_csv('/data/gumbou/codespace/data/cmcc/train.csv')
    df = df[['sentence_sep', 'label_raw']]
    df.to_csv('/data/gumbou/codespace/data/cmcc-short/train.csv', index=False)


'''
加载一篇pdf
'''
def load_pdf(pdf_path):
    from langchain.document_loaders import UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(pdf_path)
    data = loader.load()
 
    return  data

'''
加载一个目录下的pdf
'''
def load_pdf_directory(pdf_directory):
    from langchain.document_loaders import PyPDFDirectoryLoader
    loader = PyPDFDirectoryLoader(pdf_directory)
    docs = loader.load()
    return docs
    
'''
分割文本至合适的长度
'''
def to_chunks(docs,chunk_size=1000,chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_documents(docs)
    return text_chunks

'''
保存到向量库
'''
def init_vector(chunks,persist_directory,embedding_model_path):
    model_kwargs = {'device': 'cuda:0'}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceBgeEmbeddings(
                    model_name=embedding_model_path,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    query_instruction="为文本生成向量表示用于文本检索"
                )

    # model = SentenceTransformerEmbeddings(model_name=embedding_model_path)
    # embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_path,device='cuda:0',normalize_embeddings=True)
    db = Chroma.from_documents(documents=chunks, embedding=embedding,persist_directory=persist_directory)
    db.persist()
    
def get_retriver(persist_directory):
    model_name = "/home/gumbou/models/m3e-base"
    model_kwargs = {'device': 'cuda:0'}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    query_instruction="为文本生成向量表示用于文本检索"
                )
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    return vectordb
def get_clean_context(context):
    content = []
    for c in context:
        # print(c[0].page_content)
        content.append(c[0].page_content)
    return content

if __name__ == "__main__":
    # short_data()
    pdf_directory='data/wenshi_doc/'
    docs =load_pdf_directory(pdf_directory)
    # print(docs)
    
    chunks = to_chunks(docs)
    # print(chunks)
    
    persist_directory = "vector/wenshi/db-2"
    embedding_model_path='/home/gumbou/models/m3e-base'

    init_vector(chunks,persist_directory,embedding_model_path)
    
    
    # test
    # vector = get_retriver(persist_directory)
    # query = "成为股东的条件"
    # docs = vector.similarity_search_with_score(query)
    # print(docs)

