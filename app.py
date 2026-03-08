import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# Environment
# -----------------------------

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")


# -----------------------------
# Flask App
# -----------------------------

app = Flask(__name__)
CORS(app)


# -----------------------------
# Paths
# -----------------------------

DB_DIR = "db_v8"
DATA_DIR = "data"


# -----------------------------
# Embeddings
# -----------------------------


embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# Build Vector DB
# -----------------------------

def build_vector_db():

    loaders = []

    files = [
        "skills.txt",
        "profile.txt",
        "experience.txt",
        "education.txt",
        "about_me.txt",
        "projects.txt"
    ]

    for file in files:
        path = os.path.join(DATA_DIR, file)
        if os.path.exists(path):
            loaders.append(TextLoader(path))

    pdf_path = os.path.join(DATA_DIR, "resume.pdf")
    if os.path.exists(pdf_path):
        loaders.append(PyPDFLoader(pdf_path))

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_DIR
    )

    db.persist()

    print("Vector database created successfully.")

    return db


# -----------------------------
# Load Vector DB
# -----------------------------

print("Loading vector database...")

if not os.path.exists(DB_DIR):
    print("Vector DB not found. Building database...")
    db = build_vector_db()
else:
    print("Vector DB found. Loading existing database...")
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )


# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    api_key=groq_key,
    model="llama-3.3-70b-versatile",
    temperature=0
)


# -----------------------------
# Prompt
# -----------------------------

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
         "You are an AI assistant that answers questions about Ajay based ONLY on the provided context.\n"
        "Ajay Sharma Kalaga is a Python backend developer specializing in Django and Django REST Framework.\n\n"
        "Rules:\n"
        "- Answer strictly using the context below.\n"
        "- If the question asks who Ajay is, what Ajay does, his role, job, or work, summarize Ajay's professional role and responsibilities from the context instead of listing skills.\n"
        "- Use information from profile.txt or experience.txt when answering role or job related questions.\n"
        "- If the question asks about skills or technologies, organize them into clear categories such as Programming Languages, Backend Development, Databases, Data Structures & Algorithms, AI/ML, and Tools.\n"
        "- Present skills in structured sections with plain text category headings followed by a colon (for example: Programming Languages:).\n"
        "- Insert a newline before every category heading so sections are clearly separated.\n"
        "- Under each heading list items on separate lines using bullet points like • Python.\n"
        "- Each category heading must start on a new line and its bullet points must appear on separate lines below it.\n"
        "- Do NOT use markdown symbols such as ## or * for headings.\n"
        "- Keep each category separate and avoid mixing categories together.\n"
        "- Do not repeat explanatory sentences like 'Based on the provided context'. Only output the categorized skills.\n"
        "- Prefer structured lists when possible.\n"
        "- If the answer is not present in the context, say: 'That information is not available in Ajay's profile.'\n"
        "- Do not make up information.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
)


# -----------------------------
# QA Chain
# -----------------------------

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 4}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)


# -----------------------------
# API Route
# -----------------------------

@app.route("/")
def home():
    return "Ajay AI backend running"

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"response": "Invalid request"}), 400

    message = data["message"]

    result = qa.invoke({"query": message})

    return jsonify({
        "response": result["result"]
    })


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)