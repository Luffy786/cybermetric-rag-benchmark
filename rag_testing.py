# CyberMetric Benchmark Evaluation - Milestone 2
# Zero-shot vs Dense RAG vs Hybrid RAG (BM25 + Dense)
# Knowledge Base: nist_gold_kb (NIST PDFs + 30+ cybersecurity books)

import os
import json
import time
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple
import xml.etree.ElementTree as ET

import pandas as pd
import ollama
import chromadb
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi


# =========================
# CONFIGURATION
# =========================

MODELS_TO_TEST = ["qwen3:4b-instruct"]

RESULTS_DIR = "cybermetric_milestone2_nistgold_enhanced"
CYBERMETRIC_JSON_FILE = "CyberMetric-80-v1.json"

GENERATION_PARAMS = {
    "temperature": 0.1,
    "num_predict": 50,
    "top_p": 0.9,
    "repeat_penalty": 1.1
}

CHROMADB_DIR = "chromadb_store"
COLLECTION_NAME = "nist_gold_kb"
EMBED_MODEL = "all-minilm:latest"


# =========================
# HYBRID RAG SYSTEM
# =========================

class HybridRAGSystem:
    def __init__(self, persist_directory=CHROMADB_DIR, collection_name=COLLECTION_NAME):
        print("=" * 80)
        print("Initializing Hybrid RAG System (NIST Gold KB)")
        print("=" * 80)

        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        try:
            self.embedder = OllamaEmbeddings(model=EMBED_MODEL)
            ollama.show(EMBED_MODEL)
            print(f"✓ Embedding model '{EMBED_MODEL}' available")
        except Exception as e:
            print(f"ERROR: Embedding model '{EMBED_MODEL}' not available: {e}")
            self.collection = None
            return

        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            print(f"✓ Loaded collection '{collection_name}' with {self.collection.count()} chunks")
        except Exception as e:
            print(f"ERROR: Collection '{collection_name}' not found: {e}")
            self.collection = None
            return

        print("\nBuilding BM25 index...")
        self._build_bm25_index()
        print("✓ BM25 index built")
        print("=" * 80)

    def _build_bm25_index(self):
        all_results = self.collection.get()
        self.all_documents = all_results["documents"]
        self.all_metadatas = all_results["metadatas"]

        tokenized_corpus = [doc.lower().split() for doc in self.all_documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"   Indexed {len(self.all_documents)} documents")

    def dense_retrieve(self, question: str, n_results=3):
        if not self.collection:
            return [], [], []

        try:
            query_emb = self.embedder.embed_query(question)
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=n_results
            )
            return (
                results["documents"][0] if results["documents"] else [],
                results["distances"][0] if results["distances"] else [],
                results["metadatas"][0] if results["metadatas"] else []
            )
        except Exception as e:
            print(f"Dense retrieval error: {e}")
            return [], [], []

    def sparse_retrieve(self, question: str, n_results=3):
        try:
            scores = self.bm25.get_scores(question.lower().split())
            top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
            return (
                [self.all_documents[i] for i in top_ids],
                [scores[i] for i in top_ids],
                [self.all_metadatas[i] for i in top_ids]
            )
        except Exception as e:
            print(f"Sparse retrieval error: {e}")
            return [], [], []

    def hybrid_retrieve(self, question: str, n_dense=2, n_sparse=2):
        dense_docs, _, dense_meta = self.dense_retrieve(question, n_dense)
        sparse_docs, _, sparse_meta = self.sparse_retrieve(question, n_sparse)

        seen = set()
        contexts, metadatas = [], []
        info = {"dense_count": 0, "sparse_count": 0, "overlap_count": 0, "total_unique": 0}

        for d, m in zip(dense_docs, dense_meta):
            if d not in seen:
                seen.add(d)
                contexts.append(d)
                metadatas.append(m)
                info["dense_count"] += 1

        for d, m in zip(sparse_docs, sparse_meta):
            if d not in seen:
                seen.add(d)
                contexts.append(d)
                metadatas.append(m)
                info["sparse_count"] += 1
            else:
                info["overlap_count"] += 1

        info["total_unique"] = len(contexts)
        return contexts, info, metadatas


# =========================
# PROMPT / PARSING
# =========================

def make_cybermetric_prompt(question, answers, context=None):
    options = ", ".join([f"{k}) {v}" for k, v in answers.items()])

    if context:
        instructions = (
            "You are a cybersecurity expert AI assistant.\n"
            "Use the provided context if relevant.\n"
            "Choose the correct option (A, B, C, or D) only.\n"
            "Return only: <xml>answer</xml>\n\n"
            f"Context:\n{context}\n\n"
        )
    else:
        instructions = (
            "You are a cybersecurity expert AI assistant.\n"
            "Choose the correct option (A, B, C, or D) only.\n"
            "Return only: <xml>answer</xml>\n\n"
        )

    return f"{instructions}#Question: {question}\nOptions: {options}\n\nAnswer:"

def parse_xml_answer(response):
    try:
        root = ET.fromstring(response.strip())
        return root.text.strip().upper() if root.text else "PARSE_ERROR"
    except:
        pass

    try:
        start = response.find("<xml>") + 5
        end = response.find("</xml>")
        if start > 4 and end > start:
            return response[start:end].strip().upper()
    except:
        pass

    for ch in response.upper():
        if ch in ["A", "B", "C", "D"]:
            return ch

    return "PARSE_ERROR"


def generate_response(model, question, answers, context=None):
    try:
        prompt = make_cybermetric_prompt(question, answers, context)
        resp = ollama.generate(model=model, prompt=prompt, options=GENERATION_PARAMS)
        return resp["response"].strip()
    except Exception as e:
        print(f"Generation error ({model}): {e}")
        return "<xml>ERROR</xml>"


# =========================
# EVALUATION
# =========================

def evaluate_cybermetric(model, questions, rag):
    results = []
    zs_c = d_c = h_c = 0

    for i, q in enumerate(questions):
        try:
            qtext, answers, gt = q["question"], q["answers"], q["solution"]

            zs = parse_xml_answer(generate_response(model, qtext, answers))
            zs_ok = zs == gt
            zs_c += zs_ok

            d_ctx, _, d_meta = rag.dense_retrieve(qtext)
            d_prompt = "\n".join(d_ctx[:2])
            d = parse_xml_answer(generate_response(model, qtext, answers, d_prompt))
            d_ok = d == gt
            d_c += d_ok

            h_ctx, info, h_meta = rag.hybrid_retrieve(qtext)
            h_prompt = "\n".join(h_ctx[:2])
            h = parse_xml_answer(generate_response(model, qtext, answers, h_prompt))
            h_ok = h == gt
            h_c += h_ok

            results.append({
                "question_id": i,
                "correct_answer": gt,
                "zeroshot": zs,
                "dense": d,
                "hybrid": h,
                "zs_correct": int(zs_ok),
                "dense_correct": int(d_ok),
                "hybrid_correct": int(h_ok),
                "hybrid_total_contexts": info["total_unique"]
            })

            if (i + 1) % 50 == 0:
                print(f"Progress {i+1}/{len(questions)} | "
                      f"ZS {zs_c/(i+1)*100:.1f}% | "
                      f"D {d_c/(i+1)*100:.1f}% | "
                      f"H {h_c/(i+1)*100:.1f}%")

        except Exception as e:
            print(f"Error on question {i}: {e}")
            traceback.print_exc()

    total = len(questions)
    return (
        results,
        zs_c / total * 100,
        d_c / total * 100,
        h_c / total * 100
    )


# =========================
# MAIN
# =========================

def load_dataset(path):
    print(f"Loading dataset: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"] if isinstance(data, dict) else data


def main():
    print("=" * 80)
    print("CYBERMETRIC MILESTONE 2 EVALUATION")
    print("=" * 80)

    questions = load_dataset(CYBERMETRIC_JSON_FILE)
    rag = HybridRAGSystem()

    if not rag.collection:
        print("Knowledge base not available.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for model in MODELS_TO_TEST:
        print(f"\nTesting model: {model}")
        results, zs, d, h = evaluate_cybermetric(model, questions, rag)

        df = pd.DataFrame(results)
        out_csv = f"{RESULTS_DIR}/{model.replace(':','_')}_results.csv"
        df.to_csv(out_csv, index=False)

        summary = {
            "model": model,
            "zeroshot_accuracy": round(zs, 2),
            "dense_accuracy": round(d, 2),
            "hybrid_accuracy": round(h, 2),
            "timestamp": datetime.now().isoformat()
        }

        with open(out_csv.replace(".csv", ".json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Results saved for {model}")

    print("\nEvaluation completed.")

if __name__ == "__main__":
    main()
