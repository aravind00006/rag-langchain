import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents
from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.3-70b-versatile",
    ):
        """
        Initialize RAG system.
        - Loads existing FAISS index if available
        - Otherwise builds it from documents inside /data
        - Initializes Groq LLM
        """

        self.persist_dir = persist_dir

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        # Initialize vector store
        self.vectorstore = FaissVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
        )

        # Build or Load
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            print("[INFO] No existing vector store found. Building new one...")
            documents = load_all_documents("data")
            if not documents:
                raise ValueError("No documents found inside /data folder.")
            self.vectorstore.build_from_documents(documents)
        else:
            print("[INFO] Existing vector store found. Loading...")
            self.vectorstore.load()

        # Initialize LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """
        Perform similarity search and summarize results.
        """

        results = self.vectorstore.query(query, top_k=top_k)

        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r.get("metadata")
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        prompt = f"""
You are a helpful assistant.

User Query:
{query}

Relevant Context:
{context}

Provide a concise and accurate summary that answers the query using only the context above.
"""

        response = self.llm.invoke(prompt)

        return response.content.strip()
