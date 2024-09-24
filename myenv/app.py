import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs
def extract_pdf_text(pdf_files):
    text=""
    for pdf in pdf_files:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate embeddings and create a vector store
def generate_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load conversational chain for question answering
def load_qa_conversational_chain():
    # Define a template for the prompt to be used in the conversational chain
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # Initialize a chat model for generating responses
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    # Create a prompt template object with the defined template and input variables
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Load the conversational chain with the chat model and prompt template
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function for user input and generating response
def get_user_input_and_generate_response(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = load_qa_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])

# Main function for Streamlit application
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        get_user_input_and_generate_response(user_question)
    with st.sidebar:
        st.title("Menu:")
        pdf_files = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = extract_pdf_text(pdf_files)
                text_chunks = split_text_into_chunks(raw_text)
                generate_vector_store(text_chunks)
                st.success("Done")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
