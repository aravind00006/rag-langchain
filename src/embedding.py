import numpy as np
from typing import List, Any
from src.data_loader import load_all_documents
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class EmbeddingPipleline:
    def __init__(self, model_name:str = "sentence-transformers/all-MiniLM-L6-v2", chunk_size:int = 600, chunk_overlap= 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model {model_name}")
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function = len,
            separators=["\n\n","\n", ' ', '']
        )
        chunks = splitter.split_documents(documents)
        print(f'[INFO] split documents {len(documents)} into chunks: {len(chunks)}')
        return chunks
    
    def embedd_chunk(self, chunks:List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f'[INFO] Generating embeddings for {len(texts)} chunks...')
        embeddings =self.model.encode(texts, show_progress_bar=True)
        print(f'[INFO] Embeddings shape: {embeddings.shape}')
        return embeddings