# -*- coding: utf-8 -*-
"""app.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uhF5KhxOhY88wXlY-0RvQU5LI2g8K5fg
"""


import streamlit as st
import os
from langchain.output_parsers import StrOutputParser
#from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

class RAGApp:
    def __init__(self):
        if 'chain' not in st.session_state:
            st.session_state.chain = None
        if 'pdf_loaded' not in st.session_state:
            st.session_state.pdf_loaded = False

    def ensure_directory_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def load_pdf(self, pdf_path):
        try:
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()

            if not docs:
                st.error("No documents found in the PDF.")
                return None

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=2000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings()

            index_directory = "vector_store"
            index_path = os.path.join(index_directory, "faiss_index")
            self.ensure_directory_exists(index_directory)

            if os.path.exists(index_path):
                st.info(f"Loading FAISS index from {index_path}...")
                db = FAISS.load_local(index_path, embeddings)
            else:
                st.info("Creating new FAISS index...")
                db = FAISS.from_documents(texts, embeddings)
                db.save_local(index_path)

            retriever = db.as_retriever()
            
            # Define prompt template
            template = """Answer the question based only on the following context:
            {context}
            
            Question: {question}
            
            Answer: """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # Initialize model
            llm = Ollama(model="llama2")
            
            # Create chain
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            return chain

        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return None

    def run(self):
        st.title("📄 RAG PDF Chatbot")

        st.sidebar.header("📂 PDF Upload")
        uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            with open("temp_uploaded.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.chain = self.load_pdf("temp_uploaded.pdf")

            if st.session_state.chain:
                st.sidebar.success("PDF successfully loaded and processed!")
                st.session_state.pdf_loaded = True

        st.header("💬 Ask Questions about Your PDF")

        if not st.session_state.pdf_loaded:
            st.warning("Please upload a PDF first.")
        else:
            user_question = st.text_input("Enter your question:")

            if user_question and st.session_state.chain:
                with st.spinner("Generating response..."):
                    try:
                        result = st.session_state.chain.invoke(user_question)
                        st.success("Answer:")
                        st.write(result)
                    except Exception as e:
                        st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    app = RAGApp()
    app.run()
