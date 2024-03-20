import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
import streamlit as st
from langchain.chat_models import ChatOpenAI 

from langchain.embeddings import HuggingFaceBgeEmbeddings
from torch import embedding

os.environ['OPENAI_API_KEY']= "" 

# hugging face embedding

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")



# pretrained_model_name = "BAAI/bge-base-en-v1.5"
# encode_kwargs = {'normalize_embeddings': True}
# # create embeddings using hugging face bge base en v 1.5
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name = pretrained_model_name,
#     model_kwargs = {'device':'cuda:2'},
#     encode_kwargs = encode_kwargs
# )

pdf_path = 'data/LILRB2:PirB mediates macrophage recruitment in fibrogenesis of nonalcoholic steatohepatitis.pdf'
loader = PyPDFLoader(file_path=pdf_path)
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
chunk_overlap= 100, 
separators=["\n\n","\n"," ",""]) 
text = text_splitter.split_documents(documents= doc) 

# creating embeddings using huggingface bge base v 1.5

#embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(text, embeddings)
vectorstore.save_local("vectors")
print("Embeddings successfully saved in vector Database and saved locally")

# Loading the saved embeddings 
new_vector_store =FAISS.load_local("vectors", embeddings, allow_dangerous_deserialization=True)

# ConversationalRetrievalChain 
qa = ConversationalRetrievalChain.from_llm(
   llm=ChatOpenAI(model_name='gpt-3.5-turbo-0125'), 
   retriever= new_vector_store.as_retriever()
)

def document_data(query, chat_history):

   #  pdf_path = 'data/LILRB2:PirB mediates macrophage recruitment in fibrogenesis of nonalcoholic steatohepatitis.pdf'
   #  loader = PyPDFLoader(file_path=pdf_path)
   #  doc = loader.load()

   #  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
   #  chunk_overlap= 100, 
   #  separators=["\n\n","\n"," ",""]) 
   #  text = text_splitter.split_documents(documents= doc) 

   # # creating embeddings using huggingface bge base v 1.5
   
   #  #embeddings = OpenAIEmbeddings()

   #  vectorstore = FAISS.from_documents(text, embeddings)
   #  vectorstore.save_local("vectors")
   #  print("Embeddings successfully saved in vector Database and saved locally")

   # # Loading the saved embeddings 
   #  new_vector_store =FAISS.load_local("vectors", embeddings, allow_dangerous_deserialization=True)

   # # ConversationalRetrievalChain 
   #  qa = ConversationalRetrievalChain.from_llm(
   #     llm=ChatOpenAI(model_name='gpt-3.5-turbo-0125'), 
   #     retriever= new_vector_store.as_retriever()
   #  )
    
    return qa({"question":query, "chat_history":chat_history})
    

if __name__ == '__main__':

    st.header("QA ChatBot")
    # ChatInput
    prompt = st.chat_input("Enter your questions here")

    if "user_prompt_history" not in st.session_state:
       st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
       st.session_state["chat_answers_history"]=[]
    if "chat_history" not in st.session_state:
       st.session_state["chat_history"]=[]

    if prompt:
       with st.spinner("Generating......"):
           output=document_data(query=prompt, chat_history = st.session_state["chat_history"])

          # Storing the questions, answers and chat history

           st.session_state["chat_answers_history"].append(output['answer'])
           st.session_state["user_prompt_history"].append(prompt)
           st.session_state["chat_history"].append((prompt,output['answer']))

    # Displaying the chat history

    if st.session_state["chat_answers_history"]:
       for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
          message1 = st.chat_message("user")
          message1.write(j)
          message2 = st.chat_message("assistant")
          message2.write(i)
          