import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import css, bot_template, user_template
from langchain_groq.chat_models import ChatGroq
import os
import gc

# Configuration
MAX_FILE_SIZE_MB = 5
MAX_PAGES_PER_PDF = 15
MAX_FILES = 3

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            if pdf.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.warning(f"Skipped {pdf.name}: Size > {MAX_FILE_SIZE_MB}MB")
                continue

            pdf_reader = PdfReader(pdf)
            pages = pdf_reader.pages[:MAX_PAGES_PER_PDF]
            for page in pages:
                text += page.extract_text() or ""
            gc.collect()
        except Exception as e:
            st.error(f"PDF Error {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Chunking Error: {str(e)}")
        return []

def get_vectorstore(text_chunks):
    try:
        if not text_chunks:
            return None

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_texts(text_chunks, embeddings)
    except Exception as e:
        st.error(f"Embedding Error: {str(e)}")
        return None

def get_conversation_chain(vectorstore):
    if not vectorstore:
        st.error("No vector store available!")
        return None
    
    try:
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.3,
            max_tokens=512
        )

        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            max_len=3
        )

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            verbose=False
        )
    except Exception as e:
        st.error(f"LLM Error: {str(e)}")
        return None

def handle_userinput(user_question):
    if not st.session_state.get("conversation"):
        st.error("Process documents first!")
        return

    try:
        with st.spinner("Analyzing..."):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                template = user_template if i % 2 == 0 else bot_template
                st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Query Error: {str(e)}")

def main():
    load_dotenv()
    st.set_page_config(
        page_title="PDF Chat",
        page_icon="📚",
        layout="centered"
    )
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    st.session_state.setdefault("conversation", None)
    st.session_state.setdefault("chat_history", [])

    st.header("📚 PDF Chat Assistant")
    
    # Chat interface
    user_question = st.text_input("Ask about your documents:", key="user_input")
    if user_question:
        handle_userinput(user_question)

    # Processing sidebar
    with st.sidebar:
        st.subheader("Document Setup")
        pdf_docs = st.file_uploader(
            f"Upload PDFs (max {MAX_FILES})",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Max {MAX_FILE_SIZE_MB}MB per file"
        )
        
        if st.button("Process Documents"):
            st.session_state.conversation = None
            gc.collect()
            
            if not pdf_docs:
                st.warning("Upload PDFs first!")
                return

            if len(pdf_docs) > MAX_FILES:
                st.error(f"Maximum {MAX_FILES} files allowed!")
                return

            with st.status("Processing...", expanded=True) as status:
                try:
                    # Processing pipeline
                    status.write("📄 Extracting text...")
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text.strip():
                        st.error("No text found!")
                        return
                    
                    status.write("🔪 Splitting text...")
                    text_chunks = get_text_chunks(raw_text)
                    
                    if not text_chunks:
                        st.error("Failed to split text!")
                        return
                    
                    status.write("🧠 Creating knowledge base...")
                    vectorstore = get_vectorstore(text_chunks)
                    
                    if not vectorstore:
                        st.error("Failed to create vector store!")
                        return
                    
                    status.write("💡 Initializing AI...")
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    if st.session_state.conversation:
                        status.update(label="Processing complete!", state="complete")
                        st.success("Ready for questions!")
                    else:
                        st.error("AI initialization failed")
                except Exception as e:
                    st.error(f"Critical Error: {str(e)}")
                    st.stop()

        if st.button("Reset System"):
            st.session_state.clear()
            gc.collect()
            st.rerun()

if __name__ == '__main__':
    main()
