{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPRnVZZb3OALs0veovr8M79",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/cide-data/streamlit-RAG/blob/main/streamlit_RAG.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "xEKNWUuj9DsY"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "import streamlit as st\n",
        "import subprocess\n",
        "import sys\n",
        "\n",
        "# Ensure required libraries are installed\n",
        "def install_requirements():\n",
        "    try:\n",
        "        subprocess.check_call([sys.executable, '-m', 'pip', 'install',\n",
        "                                'langchain', 'langchain_community', 'langchain-openai',\n",
        "                                'scikit-learn', 'langchain-ollama', 'pymupdf',\n",
        "                                'langchain_huggingface', 'faiss-gpu', 'streamlit'])\n",
        "    except Exception as e:\n",
        "        st.error(f\"Error installing requirements: {e}\")\n",
        "\n",
        "# Import required libraries\n",
        "from langchain.chains import RetrievalQA\n",
        "from langchain_community.document_loaders import PyMuPDFLoader\n",
        "from langchain_community.llms import Ollama\n",
        "from langchain_community.vectorstores import FAISS\n",
        "from langchain_huggingface import HuggingFaceEmbeddings\n",
        "from langchain_text_splitters import CharacterTextSplitter\n",
        "\n",
        "class RAGApp:\n",
        "    def __init__(self):\n",
        "        # Initialize session state variables\n",
        "        if 'chain' not in st.session_state:\n",
        "            st.session_state.chain = None\n",
        "        if 'pdf_loaded' not in st.session_state:\n",
        "            st.session_state.pdf_loaded = False\n",
        "\n",
        "    def ensure_directory_exists(self, path):\n",
        "        \"\"\"Ensure the directory exists\"\"\"\n",
        "        if not os.path.exists(path):\n",
        "            os.makedirs(path)\n",
        "\n",
        "    def load_pdf(self, pdf_path):\n",
        "        \"\"\"Load and process PDF document\"\"\"\n",
        "        try:\n",
        "            # Load PDF\n",
        "            loader = PyMuPDFLoader(pdf_path)\n",
        "            docs = loader.load()\n",
        "\n",
        "            if not docs:\n",
        "                st.error(\"No documents found in the PDF.\")\n",
        "                return None\n",
        "\n",
        "            # Split text\n",
        "            text_splitter = CharacterTextSplitter(\n",
        "                separator=\"\\n\",\n",
        "                chunk_size=2000,\n",
        "                chunk_overlap=200\n",
        "            )\n",
        "            texts = text_splitter.split_documents(docs)\n",
        "\n",
        "            # Create embeddings\n",
        "            embeddings = HuggingFaceEmbeddings()\n",
        "\n",
        "            # Prepare vector store path\n",
        "            index_directory = \"vector_store\"\n",
        "            index_path = os.path.join(index_directory, \"faiss_index\")\n",
        "            self.ensure_directory_exists(index_directory)\n",
        "\n",
        "            # Create or load FAISS index\n",
        "            if os.path.exists(index_path):\n",
        "                st.info(f\"Loading FAISS index from {index_path}...\")\n",
        "                db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)\n",
        "            else:\n",
        "                st.info(\"Creating new FAISS index...\")\n",
        "                db = FAISS.from_documents(texts, embeddings)\n",
        "                db.save_local(index_path)\n",
        "\n",
        "            # Configure LLM\n",
        "            llm = Ollama(model=\"llama3\")\n",
        "\n",
        "            # Create retrieval QA chain\n",
        "            chain = RetrievalQA.from_chain_type(\n",
        "                llm=llm,\n",
        "                retriever=db.as_retriever()\n",
        "            )\n",
        "\n",
        "            return chain\n",
        "\n",
        "        except Exception as e:\n",
        "            st.error(f\"Error processing PDF: {e}\")\n",
        "            return None\n",
        "\n",
        "    def run(self):\n",
        "        \"\"\"Main Streamlit app\"\"\"\n",
        "        st.title(\"📄 RAG PDF Chatbot\")\n",
        "\n",
        "        # Sidebar for PDF upload\n",
        "        st.sidebar.header(\"📂 PDF Upload\")\n",
        "        uploaded_file = st.sidebar.file_uploader(\"Choose a PDF file\", type=\"pdf\")\n",
        "\n",
        "        # PDF processing\n",
        "        if uploaded_file is not None:\n",
        "            # Save uploaded file temporarily\n",
        "            with open(\"temp_uploaded.pdf\", \"wb\") as f:\n",
        "                f.write(uploaded_file.getbuffer())\n",
        "\n",
        "            # Load PDF and create chain\n",
        "            st.session_state.chain = self.load_pdf(\"temp_uploaded.pdf\")\n",
        "\n",
        "            if st.session_state.chain:\n",
        "                st.sidebar.success(\"PDF successfully loaded and processed!\")\n",
        "                st.session_state.pdf_loaded = True\n",
        "\n",
        "        # Chat interface\n",
        "        st.header(\"💬 Ask Questions about Your PDF\")\n",
        "\n",
        "        if not st.session_state.pdf_loaded:\n",
        "            st.warning(\"Please upload a PDF first.\")\n",
        "        else:\n",
        "            # Question input\n",
        "            user_question = st.text_input(\"Enter your question:\")\n",
        "\n",
        "            if user_question and st.session_state.chain:\n",
        "                with st.spinner(\"Generating response...\"):\n",
        "                    try:\n",
        "                        result = st.session_state.chain.invoke({\"query\": user_question})\n",
        "                        st.success(\"Answer:\")\n",
        "                        st.write(result['result'])\n",
        "                    except Exception as e:\n",
        "                        st.error(f\"Error generating response: {e}\")\n",
        "\n",
        "def main():\n",
        "    # Ensure requirements are installed\n",
        "    install_requirements()\n",
        "\n",
        "    # Run the Streamlit app\n",
        "    app = RAGApp()\n",
        "    app.run()\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()"
      ]
    }
  ]
}