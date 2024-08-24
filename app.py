import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.title("YouTube Transcript Query")

# Input for YouTube URL
video_url = st.text_input("Enter YouTube URL")

# Process video button
if st.button("Process Video"):
    with st.spinner("Processing video..."):
        try:
            # Load video data
            loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
            data = loader.load()

            # Check if data is loaded
            if not data:
                st.error("No data found for the provided YouTube URL. Please check the URL.")
            else:
                # Split the data into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(data)

                # Create embeddings and vector store
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                docsearch = FAISS.from_documents(docs, embeddings)
                retriever = docsearch.as_retriever()
                retriever.search_kwargs['distance_metric'] = 'cos'
                retriever.search_kwargs['k'] = 4

                # Create the QA chain
                llm = OllamaLLM(model='llama3.1')
                qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

                # Store the QA chain in session state
                st.session_state.qa = qa
                st.success("Video processed successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Input for query
query = st.text_input("Enter your query")

# Get answer button
if st.button("Get Answer"):
    if 'qa' in st.session_state:
        answer = st.session_state.qa.run(query)
        st.write(answer)
    else:
        st.error("Please process the video first.")
