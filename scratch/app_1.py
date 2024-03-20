import os
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
#from langchain.llms import OpenAI
import streamlit as st
from langchain.chat_models import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel


from langchain import hub


# Path to your key text file
key_file_path = 'key.txt'

# Read the key from the text file
with open(key_file_path, 'r') as file:
    api_key = file.read().strip()

# Set the API key as environment variable
os.environ['OPENAI_API_KEY'] = api_key

# hugging face embedding

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")





# Loading the saved embeddings 
#new_vector_store =FAISS.load_local("vectors", embeddings, allow_dangerous_deserialization=True)
new_vector_store =FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)

# # ConversationalRetrievalChain 
# qa = ConversationalRetrievalChain.from_llm(
#    llm=ChatOpenAI(model_name='gpt-3.5-turbo-0125'), 
#    retriever= new_vector_store.as_retriever()
# )

# def document_data(query, chat_history):

    
#     return qa({"question":query, "chat_history":chat_history})



# Retrieve and generate using the relevant snippets of the blog.
retriever = new_vector_store.as_retriever(search_kwargs={"k": 1})
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

 
 
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

def generate_answer(query):
   ans = rag_chain_with_source.invoke(query)
   
   return   ans

if __name__ == '__main__':

    st.header("A Query Tool for Biomedical Researchers")
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
           #output=(query=prompt, chat_history = st.session_state["chat_history"])
            output=generate_answer(prompt)
          # Seperating the metadata for reference
            ref = output['context'][0].metadata
            #st.session_state["chat_answers_history"].append(output['answer']+'\n'+ref['source'])
            answer = output['answer']
            source = ref['source']
            # display the answer with reference
            st.session_state["chat_answers_history"].append(f"{answer}\n\nThe references:\n[{source}](your_ref_here)")
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_history"].append((prompt,output['answer']))

    # Displaying the chat history

    if st.session_state["chat_answers_history"]:
       for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
          message1 = st.chat_message("user")
          message1.write(j)
          message2 = st.chat_message("assistant")
          message2.write(i)
          