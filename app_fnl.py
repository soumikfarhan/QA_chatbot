#!/usr/bin/env python
# coding: utf-8

# A Query Tool for Biomedical Researchers

# In[ ]:


# import necessary packages
import os
from langchain.vectorstores import FAISS
import streamlit as st
from langchain.chat_models import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain import hub
from langchain_community.embeddings import HuggingFaceEmbeddings


# In[ ]:


# load the open ai api key from the key.txt file

# Path to your key text file
key_file_path = 'key.txt'

# Read the key from the text file
with open(key_file_path, 'r') as file:
    api_key = file.read().strip()

# Set the API key as environment variable
os.environ['OPENAI_API_KEY'] = api_key


# In[ ]:


# load the saved embeddings from the vector store
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
new_vector_store = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)


# setups for the rag systems

# In[ ]:


# initiate the retriver to generate top results (taking top k = 1 results)
retriever = new_vector_store.as_retriever(search_kwargs={"k": 1})
# the prompt
prompt = hub.pull("rlm/rag-prompt")
# the llm
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# In[ ]:


# formatiing the docs to pass inside the chain for the context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# In[ ]:


# formulate the chain
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)


# In[ ]:


rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)


# In[ ]:


# initiate the question invoke for generating answers from llm
def generate_answer(query):
   ans = rag_chain_with_source.invoke(query)
   return   ans


# App interfacing using streamlit

# In[ ]:


if __name__ == '__main__':

    st.header("A Query Tool for Biomedical Researchers")
    # ChatInput
    prompt = st.chat_input("Enter your questions here")

   # Storing the prompts and responses history 
   
    if "user_prompt_history" not in st.session_state:
       st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
       st.session_state["chat_answers_history"]=[]
       
   # in case, you want to store the history and send it to the llm    
   #  if "chat_history" not in st.session_state:
   #     st.session_state["chat_history"]=[]

    if prompt:
       with st.spinner("Generating......"):

            output=generate_answer(prompt)
          # Seperating the metadata for reference
            ref = output['context'][0].metadata
            
            answer = output['answer']
            source = ref['source']
            # display the answer with reference
            st.session_state["chat_answers_history"].append(f"{answer}\n\nThe references:\n[{source}](your_ref_here)")
            st.session_state["user_prompt_history"].append(prompt)
            
            # in case the whole chat history can be send to the llm for conversational purposes
            # st.session_state["chat_history"].append((prompt,output['answer']))

    # Displaying the chat history for a session

    if st.session_state["chat_answers_history"]:
       for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
          message1 = st.chat_message("user")
          message1.write(j)
          message2 = st.chat_message("assistant")
          message2.write(i)

