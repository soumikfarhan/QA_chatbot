#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary packages
import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import BSHTMLLoader


# In[ ]:


def find_files(directory,extension):
    file_location = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_location.append(os.path.join(root, file))
    return file_location


# In[ ]:


def pdf_chunks(all_pdf_path,preloaded_text_splitter):
    
    all_texts = []
    
    for pdf_path in all_pdf_path:
    
    
        loader = PyPDFLoader(file_path=pdf_path)
        doc = loader.load()
        text = preloaded_text_splitter.split_documents(documents = doc)
        
        all_texts.extend(text)
    
    return all_texts


# In[ ]:


def html_chunks(all_html_path):
    
    all_html = []
    
    for html_path in all_html_path:        
        loader_bs4 = BSHTMLLoader(html_path)
        data = loader_bs4.load()
        all_html.extend(data)
    return all_html

