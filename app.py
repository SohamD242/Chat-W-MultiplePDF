import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

INDEX_PATH = "faiss_index"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(INDEX_PATH)
    st.success("FAISS index created and saved successfully.")

def get_conversational_chain(retriever):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context", and don't provide the wrong answer.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm_chain = LLMChain(llm=model, prompt=prompt)
    combine_docs_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    question_generator_prompt = PromptTemplate(template="{input_text}", input_variables=["input_text"])
    question_generator = LLMChain(llm=model, prompt=question_generator_prompt)

    chain = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        return_source_documents=True
    )

    return chain

def user_input(user_question, chat_history, retriever):
    chain = get_conversational_chain(retriever)
    
    # Convert chat history to a list of tuples (question, answer)
    formatted_chat_history = [(entry["question"], entry["answer"]) for entry in chat_history]
    
    # Call the chain with the correct argument structure
    response = chain({
        "question": user_question,
        "chat_history": formatted_chat_history
    })

    return response["answer"]

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        index_file_path = os.path.join(INDEX_PATH, "index.faiss")
        
        if not os.path.exists(index_file_path):
            st.error(f"Index file {index_file_path} does not exist. Please ensure the index is created before querying.")
            return

        new_db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever()
        
        response = user_input(user_question, st.session_state.chat_history, retriever)
        st.session_state.chat_history.append({"question": user_question, "answer": response})
        st.write("Reply: ", response)

        # Clear chat history to prevent input key mismatches
        st.session_state.chat_history = []

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)

if __name__ == "__main__":
    main()
