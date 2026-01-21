# CyberMetric Benchmark Evaluation (Zero-Shot Only)

import os
import json
import time
from datetime import datetime
from typing import List, Dict
import xml.etree.ElementTree as ET

import pandas as pd
import ollama

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_DIR = "cybermetric_zeroshot_results"
CYBERMETRIC_JSON_FILE = "CyberMetric-10000-v1.json"

MODELS_TO_TEST = [
    "qwen3-vl:4b-instruc"
]

GENERATION_PARAMS = {
    "temperature": 0.1,
    "num_predict": 50,
    "top_p": 0.9,
    "repeat_penalty": 1.1
}

# ==========================================
# PROMPT & PARSING UTILS
# ==========================================

def make_cybermetric_zeroshot_prompt(question: str, answers: Dict[str, str]) -> str:
    options_str = ', '.join([f"{key}) {value}" for key, value in answers.items()])
    
    instructions = (
        "You are a cybersecurity expert AI assistant.\n"
        "Instructions:\n"
        "a. Carefully read the question.\n"
        "b. Choose the correct answer (A, B, C, or D) only.\n"
        "c. Do NOT include any explanation or additional text.\n"
        "d. Always return the answer in this XML format: '<xml>answer</xml>'.\n\n"
    )
    
    return f"{instructions}#Question: {question}\nOptions: {options_str}\n\nAnswer:"


def parse_xml_answer(response: str) -> str:
    try:
        root = ET.fromstring(response.strip())
        if root.text:
            return root.text.strip().upper()
    except Exception:
        pass
    
    try:
        start = response.find('<xml>') + 5
        end = response.find('</xml>')
        if start > 4 and end > start:
            return response[start:end].strip().upper()
    except Exception:
        pass
    
    for char in response.strip().upper():
        if char in ['A', 'B', 'C', 'D']:
            return char
            
    return "PARSE_ERROR"


def generate_response(model_name: str, question: str, answers: Dict[str, str]) -> str:
    try:
        prompt = make_cybermetric_zeroshot_prompt(question, answers)
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options=GENERATION_PARAMS
        )
        return response['response'].strip()
    except Exception as e:
        print(f"[Inference Error] {model_name}: {e}")
        return "<xml>ERROR</xml>"


# ==========================================
# EVALUATION LOGIC
# ==========================================

def evaluate_model_zeroshot(model_name: str, questions_data: List[Dict]):
    print(f"Evaluating {model_name} (Zero-Shot)")
    
    results = []
    correct_count = 0
    total = len(questions_data)
    start_time = time.time()
    
    for i, q_data in enumerate(questions_data):
        try:
            question = q_data['question']
            answers = q_data['answers']
            correct_answer = q_data['solution']
            
            raw_response = generate_response(model_name, question, answers)
            predicted_answer = parse_xml_answer(raw_response)
            
            is_correct = int(predicted_answer == correct_answer)
            correct_count += is_correct
            
            results.append({
                "question_id": i,
                "question": question[:100],
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "raw_response": raw_response[:50]
            })
            
            if (i + 1) % max(1, total // 10) == 0:
                print(f"Progress: {i + 1}/{total} | "
                      f"Accuracy: {correct_count/(i+1)*100:.2f}%")
        
        except Exception as e:
            print(f"[Evaluation Error] Question {i}: {e}")

    accuracy = (correct_count / total) * 100 if total > 0 else 0
    duration = time.time() - start_time
    
    return results, accuracy, duration


def save_results(model_name: str, results: List[Dict], accuracy: float, duration: float):
    df_results = pd.DataFrame(results)
    safe_name = model_name.replace(':', '_').replace('/', '_')
    detail_file = f"{RESULTS_DIR}/{safe_name}_detailed.csv"
    df_results.to_csv(detail_file, index=False)
    
    leaderboard_file = f"{RESULTS_DIR}/_LEADERBOARD.csv"
    new_entry = {
        "model": model_name,
        "accuracy": round(accuracy, 2),
        "questions_n": len(results),
        "time_sec": round(duration, 1),
        "timestamp": datetime.now().isoformat()
    }
    
    if os.path.exists(leaderboard_file):
        df_leaderboard = pd.read_csv(leaderboard_file)
        df_leaderboard = df_leaderboard[df_leaderboard['model'] != model_name]
        df_leaderboard = pd.concat(
            [df_leaderboard, pd.DataFrame([new_entry])],
            ignore_index=True
        )
    else:
        df_leaderboard = pd.DataFrame([new_entry])
        
    df_leaderboard = df_leaderboard.sort_values(by="accuracy", ascending=False)
    df_leaderboard.to_csv(leaderboard_file, index=False)
    
    print(f"Results saved for {model_name} | Accuracy: {accuracy:.2f}%")


def load_cybermetric_dataset(json_file: str) -> List[Dict]:
    print(f"Loading dataset from {json_file}")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "questions" in data:
            return data["questions"]
        elif isinstance(data, list):
            return data
        else:
            print("Dataset format not recognized")
            return []
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return []


def main():
    print("=" * 60)
    print("CYBERMETRIC ZERO-SHOT BENCHMARK")
    print("=" * 60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    questions_data = load_cybermetric_dataset(CYBERMETRIC_JSON_FILE)
    if not questions_data:
        return

    try:
        local_models = [m['name'] for m in ollama.list()['models']]
    except Exception:
        local_models = []

    for model in MODELS_TO_TEST:
        print(f"\n{'='*20} MODEL: {model} {'='*20}")
        
        if local_models and not any(model in m for m in local_models):
            print(f"Note: '{model}' not found locally. Ollama may pull it automatically.")

        results, accuracy, duration = evaluate_model_zeroshot(model, questions_data)
        save_results(model, results, accuracy, duration)

    print("\nBenchmark completed successfully.")
    print(f"Results available in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
