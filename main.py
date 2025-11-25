import json
import re
import requests
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

import puzzle_gen

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:14b"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# LOAD PUZZLES DYNAMICALLY
CHESS_PUZZLES = puzzle_gen.get_puzzles(csv_path="lichess_db_puzzle.csv", samples_per_category=5)

@dataclass
class CoTResponse:
    puzzle_id: str
    fen: str
    difficulty: str
    prompt: str
    cot: str
    timestamp: float
    rating: int
    
@dataclass
class BehavioralMetrics:
    puzzle_id: str
    difficulty: str
    rating: int
    cot_length: int
    cot_length_tokens: int
    revision_count: int
    hedging_count: int
    loop_score: float
    progress_score: float
    convergence: bool
    appears_correct: bool

def generate_cot(prompt: str, temperature: float = 0.7) -> str:
    # Use Chat endpoint
    api_url = OLLAMA_API.replace("/api/generate", "/api/chat")
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            # Stateless: New conversation for every request
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "stream": True,
        "options": {
            "num_predict": -1, # Infinite generation limit
        }
    }
    
    current_mode = None
    
    try:
        response = requests.post(api_url, json=payload, stream=True, timeout=None)
        response.raise_for_status()
        
        full_text = ""
        
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_response = json.loads(line)
                    
                    if "error" in json_response:
                        print(f"\n\033[91mOllama Error: {json_response['error']}\033[0m")
                        break
                    
                    if "message" in json_response:
                        msg = json_response["message"]
                        
                        # Handle Thinking (CoT)
                        if "thinking" in msg and msg["thinking"]:
                            if current_mode != "thinking":
                                print(f"\n\n\033[94m=== CHAIN OF THOUGHT ===\033[0m\n", flush=True)
                                current_mode = "thinking"
                            
                            chunk = msg["thinking"]
                            print(chunk, end="", flush=True)
                            full_text += chunk
                        
                        # Handle Content (Answer)
                        if "content" in msg and msg["content"]:
                            if current_mode != "content":
                                print(f"\n\n\033[92m=== FINAL ANSWER ===\033[0m\n", flush=True)
                                current_mode = "content"
                                
                            chunk = msg["content"]
                            print(chunk, end="", flush=True)
                            full_text += chunk
                    
                    if json_response.get("done", False):
                        break
                        
                except json.JSONDecodeError:
                    continue
        
        print("\n")
        return full_text

    except Exception as e:
        print(f"\nError generating CoT: {e}")
        return ""
    
def count_revisions(cot: str) -> int:
    revision_patterns = [
        r'\bwait\b.*\bno\b',
        r'\bactually\b',
        r'\bi was wrong\b',
        r'\bon second thought\b',
        r'\blet me reconsider\b',
        r'\bno,? that\'?s? (?:not |in)?correct\b',
        r'\bI made (?:a |an )?(?:error|mistake)\b',
        r'\bcorrection\b',
    ]
    
    count = 0
    cot_lower = cot.lower()
    for pattern in revision_patterns:
        count += len(re.findall(pattern, cot_lower))
    return count

def count_hedging(cot: str) -> int:
    hedging_patterns = [
        r'\bmaybe\b',
        r'\bpossibly\b',
        r'\bperhaps\b',
        r'\bi think\b',
        r'\bi\'m not sure\b',
        r'\buncertain\b',
        r'\bmight be\b',
        r'\bcould be\b',
        r'\bprobably\b',
    ]
    
    count = 0
    cot_lower = cot.lower()
    for pattern in hedging_patterns:
        count += len(re.findall(pattern, cot_lower))
    return count

def calculate_loop_score(cot: str) -> float:
    sentences = [s.strip() for s in re.split(r'[.!?]+', cot) if len(s.strip()) > 10]
    
    if len(sentences) < 3:
        return 0.0
    
    max_similarity = 0.0
    for i, sent1 in enumerate(sentences):
        for j, sent2 in enumerate(sentences):
            if abs(i - j) < 2:
                continue
            
            words1 = set(sent1.lower().split())
            words2 = set(sent2.lower().split())
            if len(words1) == 0 or len(words2) == 0:
                continue
                
            similarity = len(words1 & words2) / max(len(words1), len(words2))
            max_similarity = max(max_similarity, similarity)
    
    return max_similarity

def calculate_progress_score(cot: str) -> float:
    stages = {
        'parse': [r'\bfen\b', r'\bparse\b', r'\bboard\b', r'\bposition\b'],
        'understand': [r'\bunderstand\b', r'\banalyze\b', r'\bpieces?\b', r'\bmaterial\b'],
        'find_moves': [r'\bmoves?\b', r'\bcandidates?\b', r'\boptions?\b'],
        'evaluate': [r'\bevaluat\w*\b', r'\bbest\b', r'\bchoose\b', r'\bdecide\b'],
    }
    
    cot_lower = cot.lower()
    stages_present = 0
    
    for stage_patterns in stages.values():
        for pattern in stage_patterns:
            if re.search(pattern, cot_lower):
                stages_present += 1
                break
    
    return stages_present / len(stages)

def check_convergence(cot: str) -> bool:
    conclusion_patterns = [
        r'\b(?:therefore|thus|so|in conclusion)\b',
        r'\bthe (?:best|correct|right) (?:move|answer) is\b',
        r'\bi (?:would|will|should) (?:play|choose|recommend)\b',
        r'\bfinal (?:answer|move|decision)\b',
    ]
    
    cot_lower = cot.lower()
    for pattern in conclusion_patterns:
        if re.search(pattern, cot_lower):
            return True
    return False

def compute_behavioral_metrics(puzzle, cot: str) -> BehavioralMetrics:
    return BehavioralMetrics(
        puzzle_id=puzzle['id'],
        difficulty=puzzle['difficulty'],
        rating=puzzle['rating'],
        cot_length=len(cot),
        cot_length_tokens=len(cot.split()),
        revision_count=count_revisions(cot),
        hedging_count=count_hedging(cot),
        loop_score=calculate_loop_score(cot),
        progress_score=calculate_progress_score(cot),
        convergence=check_convergence(cot),
        appears_correct=False,
    )

# UPDATED: Removes description/themes from the actual prompt
def create_puzzle_prompt(fen: str) -> str:
    return f"""You are analyzing a chess position. Think step by step.

FEN notation: {fen}

Task: Find the best move.

Please:
1. Parse and understand the position from the FEN notation
2. Analyze the position 
3. Find candidate moves
4. Recommend the best move and explain why

Think through this carefully, step by step."""

def run_baseline_characterization():
    print("=" * 80)
    print("PHASE 2: BASELINE CHARACTERIZATION (LICHESS DATASET)")
    print("=" * 80)
    
    all_responses = []
    all_metrics = []
    
    puzzle_count = 0
    
    for difficulty, puzzles in CHESS_PUZZLES.items():
        print(f"\n{difficulty.upper()} PUZZLES:")
        print("-" * 40)
        
        for i, puzzle in enumerate(puzzles):
            puzzle_count += 1
            
            # We still print the description for the user to see in logs,
            # but we do NOT pass it to the prompt generator.
            rating_str = f"(Rating: {puzzle['rating']})" if puzzle['rating'] > 0 else "(Broken/Empty)"
            print(f"\n[{puzzle_count}] ID: {puzzle['id']} {rating_str}")
            print(f"    (Hidden from model) Theme: {puzzle['description'][:70]}...")
            
            # NO DESCRIPTION PASSED HERE
            prompt = create_puzzle_prompt(puzzle['fen'])
            
            print("\n" + "="*20 + " PROMPT " + "="*20)
            print(prompt)
            print("="*48 + "\n")
            
            print("  Generating CoT...")
            print("-" * 20 + " STREAM START " + "-" * 20)
            
            start_time = time.time()
            cot = generate_cot(prompt)
            elapsed = time.time() - start_time
            
            print("-" * 20 + " STREAM END " + "-" * 20)
            print(f"  Time: {elapsed:.1f}s")
            
            response = CoTResponse(
                puzzle_id=puzzle['id'],
                fen=puzzle['fen'],
                difficulty=difficulty,
                prompt=prompt,
                cot=cot,
                timestamp=time.time(),
                rating=puzzle['rating']
            )
            all_responses.append(response)
            
            metrics = compute_behavioral_metrics(puzzle, cot)
            all_metrics.append(metrics)
            
            print(f"  CoT length: {metrics.cot_length_tokens} tokens")
            print(f"  Revisions: {metrics.revision_count}, Hedging: {metrics.hedging_count}")
            print(f"  Loop score: {metrics.loop_score:.2f}")
            
            time.sleep(1.0) 
    
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    responses_file = OUTPUT_DIR / "cot_responses.jsonl"
    with open(responses_file, 'w') as f:
        for resp in all_responses:
            f.write(json.dumps(asdict(resp)) + '\n')
    print(f"✓ Saved {len(all_responses)} responses to {responses_file}")
    
    metrics_file = OUTPUT_DIR / "behavioral_metrics.jsonl"
    with open(metrics_file, 'w') as f:
        for m in all_metrics:
            f.write(json.dumps(asdict(m)) + '\n')
    print(f"✓ Saved {len(all_metrics)} metrics to {metrics_file}")
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS BY DIFFICULTY")
    print("=" * 80)
    
    for difficulty in ["easy", "medium", "hard", "impossible"]:
        diff_metrics = [m for m in all_metrics if m.difficulty == difficulty]
        if not diff_metrics:
            continue
            
        print(f"\n{difficulty.upper()}:")
        print(f"  Count: {len(diff_metrics)}")
        print(f"  Avg CoT length: {np.mean([m.cot_length_tokens for m in diff_metrics]):.1f} tokens")
        print(f"  Avg revisions: {np.mean([m.revision_count for m in diff_metrics]):.2f}")
        print(f"  Avg hedging: {np.mean([m.hedging_count for m in diff_metrics]):.2f}")
        print(f"  Avg loop score: {np.mean([m.loop_score for m in diff_metrics]):.3f}")
    
    return all_responses, all_metrics

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ANXIETY MEDS FOR REASONING MODELS")
    print("Phase 1 & 2: Dataset Construction + Baseline Characterization")
    print("=" * 80)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    if not CHESS_PUZZLES:
        print("\n\033[91mCRITICAL ERROR: No puzzles loaded. Check lichess_db_puzzle.csv\033[0m")
    else:
        print(f"Total puzzles: {sum(len(p) for p in CHESS_PUZZLES.values())}")
        responses, metrics = run_baseline_characterization()
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)