import os
import streamlit as st
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.vectorstores import VectorStoreRetriever

os.environ['OPENAI_API_KEY'] = ""

# Load PDF
# pdf_path = 'path/to/the/.pdf'
# loader = PyPDFLoader(file_path=pdf_path)
# doc = loader.load()

# # Split text
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
# text = text_splitter.split_documents(documents=doc)

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# # Create and save vector store
# vectorstore = FAISS.from_documents(text, embeddings)
# vectorstore.save_local("vectors")
# print("Embeddings successfully saved in vector Database and saved locally")

# Load saved embeddings
new_vector_store = FAISS.load_local("vectors", embeddings, allow_dangerous_deserialization=True)

# Custom retriever with metadata support
class MetadataVectorStoreRetriever(VectorStoreRetriever):
    def retrieve(self, query, metadata=None, **kwargs):
        if metadata is not None:
            self.metadata = metadata
        return super().retrieve(query, **kwargs)

# Build ConversationalRetrievalChain with custom retriever
qa = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model_name='gpt-3.5-turbo-0125'), 
                                           retriever=MetadataVectorStoreRetriever(new_vector_store))

# Define document_data function
def document_data(query, chat_history):
    metadata = [meta for _, meta in chat_history]  # Extract metadata from chat history
    retriever_response = qa.retrieve(query=query, metadata=metadata)  # Query retriever with metadata
    qa_response = qa(llm_input={"question": query, "chat_history": chat_history}, retriever_output=retriever_response)  # Perform QA
    return qa_response

# Main function
if __name__ == '__main__':
    st.header("QA ChatBot")

    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []
    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    prompt = st.chat_input("Enter your questions here")

    if prompt:
        with st.spinner("Generating......"):
            output = document_data(query=prompt, chat_history=st.session_state["chat_history"])

            st.session_state["chat_answers_history"].append(output['answer'])
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_history"].append((prompt, output['answer']))

    if st.session_state["chat_answers_history"]:
        for i, j in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
            message1 = st.chat_message("user")
            message1.write(j)
            message2 = st.chat_message("assistant")
            message2.write(i)