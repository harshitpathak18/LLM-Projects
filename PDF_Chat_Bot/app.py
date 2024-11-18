import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader

# Load environment variables from the .env file
load_dotenv()

# Check for Google API key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Styling for the Streamlit page
def style_page():
    """
    Customize the layout and appearance of the Streamlit page. 
    Hides elements like the menu, footer, and header for a clean UI.
    """
    st.set_page_config(
        page_title="PDF QueryBot",
        page_icon="",
        layout="wide")

    hide_st_style = """
            <style>
            [data-testid="stAppViewContainer"] {
                background-image: url('https://digitalsynopsis.com/wp-content/uploads/2017/07/beautiful-color-ui-gradients-backgrounds-royal.png');
                background-size: cover;
                margin-top: 0px; /* Set margin-top to 0px to remove space */
            }

            #MainMenu {visibility: hidden;}
            footer{visibility: hidden;}
            header {visibility: hidden;}
            .st-emotion-cache-18ni7ap {visibility: hidden;}
            .st-emotion-cache-1avcm0n.ezrtsby2{visibility: hidden;}
            .st-emotion-cache-z5fcl4 {width: 100%; padding: 0rem 1rem 1rem;} 
            .st-emotion-cache-10trblm{text-align: center;}
            .st-emotion-cache-qdbtli {
                width: 100%;
                padding: 5px 5px 5px 5px;
                n-width: auto;
                max-width: initial;
                }
            .st-emotion-cache-kgpedg{
                padding: 0rem 0rem 0rem;
            }
            .st-emotion-cache-1gwvy71 {
                padding: 5px;
            }

            .st-emotion-cache-6qob1r{
                background-color: rgba(8, 22, 40, 0.8);
            }
            .st-emotion-cache-7tauuy{padding: 3rem 1rem 1rem;}

            .st-emotion-cache-1jicfl2 {
                width: 100%;
                padding: 5px 5px 5px 5px;
                min-width: auto;
                max-width: initial;
            }
            
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    """
    Extracts text content from uploaded PDF files.
    """
    try:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                if page is not None:
                    text += page.extract_text()
        if not text.strip():
            raise ValueError("The uploaded PDF(s) contain no readable text.")
        return text
    except Exception as e:
        st.error(f"Error reading PDF files: {e}")
        return ""

def get_text_chunks(text):
    """
    Split the extracted text into smaller chunks for processing.
    """
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
        return splitter.split_text(text)
    except Exception as e:
        st.error(f"Error splitting text into chunks: {e}")
        return []

def get_vector_store(chunks):
    """
    Generate a FAISS vector store from text chunks.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

def user_input(user_question):
    """
    Handles user input and searches the vector store for an answer.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if not os.path.exists("faiss_index"):
            raise FileNotFoundError("Vector store not found. Please process PDFs first.")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response
    except Exception as e:
        st.error(f"Error generating a response: {e}")
        return {"output_text": "Sorry, an error occurred while processing your question."}

def get_conversational_chain():
    """
    Creates a QA chain using Langchain and Google Generative AI.
    """
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context."
        Context: {context}
        Question: {question}
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.5)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        st.error(f"Error initializing conversational chain: {e}")
        return None

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me any question."}]

def main():
    """
    Main function to handle the Streamlit app.
    """
    st.title("PDF QueryBot")
    st.write("Effortless Conversations with Your Documents!")

    # File uploader with exception handling
    try:
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:
                    return
                text_chunks = get_text_chunks(raw_text)
                if not text_chunks:
                    return
                get_vector_store(text_chunks)
                st.success("Processing done. You can now ask questions!")
    except Exception as e:
        st.error(f"Error during file upload or processing: {e}")

    # Chatbot functionality
    if os.path.exists("faiss_index"):
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Ask me any question."}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Generating Answer..."):
                    response = user_input(prompt)
                    answer = response.get("output_text", "Unable to generate a response.")
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

        st.button("Clear Chat History", on_click=clear_chat_history)
    else:
        st.info("Please upload and process PDFs before asking questions.")

if __name__ == "__main__":
    style_page()
    main()
