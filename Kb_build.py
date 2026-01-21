import os
import re
import time
from typing import List

# --- CRITICAL IMPORTS ---
import fitz  # PyMuPDF (pip install pymupdf)
import pytesseract
from pdf2image import convert_from_path
import chromadb
import ollama  # CRITICAL FIX: Explicit import for the client
from langchain_ollama import OllamaEmbeddings

# ==========================================
# CONFIGURATION
# ==========================================
# Exact filenames of the NIST PDFs you uploaded
CYBERSECURITY_PDF_FILES = [
    "FIPS_140-2.pdf",
    "SP_1299.pdf",
    "SP_1300.pdf",
    "SP_1302.pdf",
    "SP_1303.pdf",
    "SP_1305.pdf",
    

    "SP_1800-12.pdf",
    "SP_1800-13.pdf",
    "SP_1800-14.pdf",
    "SP_1800-25.pdf",
    "SP_1800-26.pdf",
     "SP_800-63-4.pdf",
"SP_800-63A-4.pdf",
"SP_800-63B-4.pdf",
"SP_800-63C-4.pdf",
"SP_800-76-2.pdf",
"SP_800-78-5.pdf",
"SP_800-125B.pdf",
"SP_800-128.pdf",
"SP_800-130.pdf",
"SP_800-140.pdf",
"SP_800-152.pdf",
"SP_800-185.pdf",
"SP_800-192.pdf",
"SP_800-204.pdf",
"SP_800-204A.pdf",
"SP_800-216.pdf",
"SP_800-218.pdf",
"SP_800-218A.pdf",
"SP_800-228.pdf",
"SP_800-231.pdf",

   



]

# Settings for "Gold Standard" KB
MIN_TEXT_THRESHOLD = 500
EMBED_MODEL = "nomic-embed-text" 
COLLECTION_NAME = "nist_gold_kb"
CHROMADB_DIR = "chromadb_store"

class HybridPdfLoader:
    def __init__(self, persist_directory=CHROMADB_DIR):
        print("=" * 60)
        print("Initializing ChromaDB with NIST Gold Standard Config...")
        print("=" * 60)
        
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Verify Ollama connection
        try:
            # We use the raw client to check connectivity first
            ollama.show(EMBED_MODEL)
            print(f"✓ Ollama Model '{EMBED_MODEL}' is available.")
            self.embedder = OllamaEmbeddings(model=EMBED_MODEL)
        except Exception as e:
            print(f"CRITICAL ERROR: Ollama model '{EMBED_MODEL}' is not running or pulled.")
            print("Run: ollama pull nomic-embed-text")
            raise e 

        try:
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
            print(f"Loaded existing collection '{COLLECTION_NAME}' with {self.collection.count()} documents")
        except:
            self.collection = self.chroma_client.create_collection(COLLECTION_NAME)
            print(f"Created new ChromaDB collection: '{COLLECTION_NAME}'")

    def extract_text_pymupdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            n_pages = len(doc)
            for page_num in range(n_pages):
                text = doc[page_num].get_text("text")
                if text:
                    # Clean up excessive whitespace/formatting
                    text = re.sub(r'\s+', ' ', text)
                    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
                    full_text += text + " "
            doc.close()
            return full_text.strip()
        except Exception as e:
            print(f"    PyMuPDF failed: {e}")
            return ""

    def extract_text_ocr(self, pdf_path: str) -> str:
        try:
            print(f"    Starting OCR extraction (fallback)...")
            images = convert_from_path(pdf_path, dpi=200, fmt='jpeg')
            full_text = ""
            for i, img in enumerate(images):
                try:
                    page_text = pytesseract.image_to_string(img, lang='eng')
                    if page_text.strip():
                        page_text = re.sub(r'\s+', ' ', page_text)
                        full_text += page_text + " "
                except Exception:
                    continue
            return full_text.strip()
        except Exception as e:
            print(f"    OCR extraction failed: {e}")
            return ""

    def smart_extract_text(self, pdf_path: str) -> str:
        print(f"Extracting text from {pdf_path}...")
        text = self.extract_text_pymupdf(pdf_path)
        if len(text) >= MIN_TEXT_THRESHOLD:
            print(f"     PyMuPDF extracted {len(text)} characters")
            return text
        else:
            print(f"    ⚠ PyMuPDF text too short ({len(text)} chars). Attempting OCR...")
            ocr_text = self.extract_text_ocr(pdf_path)
            if len(ocr_text) > len(text):
                print(f"     OCR extracted {len(ocr_text)} characters")
                return ocr_text
            return text

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """
        Improved chunking for NIST documents. 
        Uses larger chunks (800 chars) to capture full policy definitions.
        """
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        start = 0
        total_len = len(text)
        
        print(f"    Chunking {total_len} characters...")
        
        while start < total_len:
            end = min(start + chunk_size, total_len)
            
            # Smart boundary detection: Try to break at a sentence ending
            last_punct = max(
                text.rfind('.', start, end),
                text.rfind('!', start, end),
                text.rfind('?', start, end)
            )
            
            # Only use punctuation if it's not too close to the start (avoid tiny chunks)
            if last_punct > start + (chunk_size // 2) and last_punct < end:
                end = last_punct + 1
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 50:  # Filter out tiny garbage chunks
                chunks.append(chunk)
                
            start = end - overlap if end < total_len else end
            if start < 0: start = 0
            
        print(f"    Created {len(chunks)} chunks.")
        return chunks

    def add_pdf_to_chromadb(self, pdf_path: str):
        full_text = self.smart_extract_text(pdf_path)
        if not full_text or len(full_text) < 100:
            print(f"    ✗ Skipping {pdf_path} (insufficient text)")
            return

        chunks = self.chunk_text(full_text)
        if not chunks:
            return

        print(f"    Embedding {len(chunks)} chunks...")
        
        current_count = self.collection.count()
        pdf_name = os.path.basename(pdf_path)
        
        # Prepare batch data
        ids = [f"{pdf_name}_{current_count + i}" for i in range(len(chunks))]
        metadatas = [{"source": pdf_name, "chunk_id": i} for i in range(len(chunks))]
        
        # Batch processing to prevent timeouts
        batch_size = 50 
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            
            try:
                # Embedding happens here automatically via LangChain wrapper
                self.collection.add(
                    documents=batch_chunks,
                    ids=batch_ids,
                    metadatas=batch_meta
                )
                print(f"     Added batch {i//batch_size + 1}")
            except Exception as e:
                print(f"     Batch failed: {e}")

    # --- THIS METHOD IS NOW CORRECTLY INDENTED INSIDE THE CLASS ---
    def process_all_pdfs(self, pdf_files: List[str]):
        print(f"\n{'=' * 60}")
        print(f"PROCESSING {len(pdf_files)} PDF FILES")
        print(f"{'=' * 60}")
        
        successful = 0
        failed = 0
        total_start = time.time()
        
        for i, pdf_file in enumerate(pdf_files):
            print(f"\n[{i + 1}/{len(pdf_files)}] Processing: {pdf_file}")
            print("-" * 50)
            
            if os.path.exists(pdf_file):
                try:
                    self.add_pdf_to_chromadb(pdf_file)
                    successful += 1
                except Exception as e:
                    print(f"     FAILED processing {pdf_file}: {e}")
                    failed += 1
            else:
                print(f"     File not found: {pdf_file}")
                failed += 1
                
        total_docs = self.collection.count()
        total_time = time.time() - total_start
        
        print(f"\n{'=' * 60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Processed: {successful}/{len(pdf_files)}")
        print(f"Total chunks in KB: {total_docs}")
        print(f"{'=' * 60}")

if __name__ == "__main__":
    loader = HybridPdfLoader()

    loader.process_all_pdfs(CYBERSECURITY_PDF_FILES)
