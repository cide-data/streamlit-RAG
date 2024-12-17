    import os
    import streamlit as st
    import subprocess
    import sys

#Ensure required libraries are installed

    def install_requirements():
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                                    'langchain', 'langchain_community', 'langchain-openai',
                                    'scikit-learn', 'langchain-ollama', 'pymupdf', 
                                    'langchain_huggingface', 'faiss-gpu', 'streamlit'])
        except Exception as e:
            st.error(f"Error installing requirements: {e}")

#Import required libraries

    from langchain.chains import RetrievalQA
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_community.llms import Ollama
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import CharacterTextSplitter
    
    class RAGApp:
        def __init__(self):
            #Initialize session state variables
            if 'chain' not in st.session_state:
                st.session_state.chain = None
            if 'pdf_loaded' not in st.session_state:
                st.session_state.pdf_loaded = False
    
        def ensure_directory_exists(self, path):
            """Ensure the directory exists"""
            if not os.path.exists(path):
                os.makedirs(path)
    
        def load_pdf(self, pdf_path):
            """Load and process PDF document"""
            try:
                #Load PDF
                loader = PyMuPDFLoader(pdf_path)
                docs = loader.load()
    
                if not docs:
                    st.error("No documents found in the PDF.")
                    return None
    
                #Split text
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=2000,
                    chunk_overlap=200
                )
                texts = text_splitter.split_documents(docs)
    
                #Create embeddings
                embeddings = HuggingFaceEmbeddings()
    
                #Prepare vector store path
                index_directory = "vector_store"
                index_path = os.path.join(index_directory, "faiss_index")
                self.ensure_directory_exists(index_directory)
    
                #Create or load FAISS index
                if os.path.exists(index_path):
                    st.info(f"Loading FAISS index from {index_path}...")
                    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
                else:
                    st.info("Creating new FAISS index...")
                    db = FAISS.from_documents(texts, embeddings)
                    db.save_local(index_path)
    
                #Configure LLM
                llm = Ollama(model="llama3")
    
                #Create retrieval QA chain
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=db.as_retriever()
                )
    
                return chain
    
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                return None
    
        def run(self):
            """Main Streamlit app"""
            st.title("📄 RAG PDF Chatbot")
    
            #Sidebar for PDF upload
            st.sidebar.header("📂 PDF Upload")
            uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    
            #PDF processing
            if uploaded_file is not None:
                #Save uploaded file temporarily
                with open("temp_uploaded.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                #Load PDF and create chain
                st.session_state.chain = self.load_pdf("temp_uploaded.pdf")
                
                if st.session_state.chain:
                    st.sidebar.success("PDF successfully loaded and processed!")
                    st.session_state.pdf_loaded = True
    
            #Chat interface
            st.header("💬 Ask Questions about Your PDF")
            
            if not st.session_state.pdf_loaded:
                st.warning("Please upload a PDF first.")
            else:
                #Question input
                user_question = st.text_input("Enter your question:")
                
                if user_question and st.session_state.chain:
                    with st.spinner("Generating response..."):
                        try:
                            result = st.session_state.chain.invoke({"query": user_question})
                            st.success("Answer:")
                            st.write(result['result'])
                        except Exception as e:
                            st.error(f"Error generating response: {e}")
    
    def main():
        #Ensure requirements are installed
        install_requirements()
        
        #Run the Streamlit app
        app = RAGApp()
        app.run()
    
    if __name__ == "__main__":
        main()
