from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str) ->List[Any]:
    """
    Load all supported files from the data directory and convert to Langchain document structure
    supported: PDf, Txt
    """
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f'[DEBUG] Data path: {data_path}')
    documents = []

    #PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f'[DEBUG] found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}')
    for pdf_file in pdf_files:
        print(f'[DEBUG] Loading PDF: {pdf_file}')
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            print(f"[DEBUG] Loader {len(docs)} PDF docs from {pdf_file}")
            documents.extend(docs)
        except Exception as e:
            print(f'[ERROR] Failed to load PDF {pdf_file}: {e}')

    return documents