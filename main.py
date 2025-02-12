import os
from apikey import apikey
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set the API key
os.environ["OPENAI_API_KEY"] = apikey

def setup_qa_system(file_path):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        retriever = vector_store.as_retriever()
        llm = ChatOpenAI(temperature=0, model_name='gpt-4')
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

        return qa_chain
    except Exception as e:
        st.error(f"Error setting up QA system: {str(e)}")
        return None

def main():
    st.title("PDF Question Answering System")
    
    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'history' not in st.session_state:
        st.session_state.history = []

    # File uploader
    uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Initialize QA system if not already initialized
        if st.session_state.qa_chain is None:
            with st.spinner("Processing PDF..."):
                st.session_state.qa_chain = setup_qa_system("temp.pdf")
        
        # Question input
        question = st.text_input("Ask a question about your PDF:")
        
        if question:
            with st.spinner("Processing question..."):
                try:
                    # Get the answer
                    answer = st.session_state.qa_chain.invoke({"query": question})
                    
                    # Add to history (prepend to show most recent first)
                    st.session_state.history.insert(0, {
                        "question": question,
                        "answer": answer['result']
                    })
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
        
        # Display history
        if st.session_state.history:
            st.subheader("Chat History")
            for item in st.session_state.history:  # No need to reverse, already in correct order
                with st.container():
                    st.write("Q:", item["question"])
                    st.write("A:", item["answer"])
                    st.write("---")
        
        # Clean up
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")

if __name__ == '__main__':
    main()