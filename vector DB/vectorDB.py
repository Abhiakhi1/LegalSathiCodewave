import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document

input_folder = r"C:\Hackathon\LegalSathi\vector DB\data"
output_folder = r"C:\Hackathon\LegalSathi\vector DB\vector_dbs"
os.makedirs(output_folder, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_text(text)

        docs = [Document(page_content=chunk, metadata={"source": filename}) for chunk in chunks]

        vectorstore = FAISS.from_documents(docs, embeddings)

        file_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
        os.makedirs(file_output_folder, exist_ok=True)
        vectorstore.save_local(file_output_folder)

        print(f"✅ Saved vector DB for {filename} at {file_output_folder}")