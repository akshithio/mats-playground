import json
import re
import requests
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
import numpy as np

import puzzle_gen

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:14b"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# LOAD PUZZLES DYNAMICALLY
CHESS_PUZZLES = puzzle_gen.get_puzzles(csv_path="lichess_db_puzzle.csv", samples_per_category=5)

@dataclass
class TokenLogProb:
    token: str
    logprob: float
    prob: float
    top_logprobs: List[Dict[str, any]] = field(default_factory=list)

@dataclass
class CoTResponse:
    puzzle_id: str
    fen: str
    difficulty: str
    prompt: str
    cot: str
    timestamp: float
    rating: int
    token_logprobs: List[TokenLogProb] = field(default_factory=list)
    
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
    avg_entropy: float = 0.0
    high_entropy_tokens: int = 0

def calculate_entropy(top_logprobs: List[Dict]) -> float:
    """Calculate Shannon entropy from token probabilities"""
    if not top_logprobs:
        return 0.0
    
    # Extract logprob values from list of dicts
    # Format: [{"token": "X", "logprob": -0.5}, {"token": "Y", "logprob": -1.2}, ...]
    logprob_values = [item["logprob"] for item in top_logprobs]
    
    # Convert logprobs to probabilities
    probs = np.exp(logprob_values)
    
    # Normalize to ensure sum = 1
    probs = probs / np.sum(probs)
    
    # Calculate entropy: H = -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy

def get_color_for_entropy(entropy: float, max_entropy: float = 3.0) -> str:
    """
    Return ANSI color code based on entropy level:
    - Low entropy (confident): Green
    - Medium entropy (uncertain): Yellow
    - High entropy (very uncertain): Red
    """
    normalized = entropy / max_entropy
    
    if normalized < 0.33:
        return "\033[92m"  # Green
    elif normalized < 0.67:
        return "\033[93m"  # Yellow
    else:
        return "\033[91m"  # Red

def generate_cot(prompt: str, temperature: float = 0.7) -> tuple[str, List[TokenLogProb]]:
    """Generate CoT with logprobs capture using /api/generate endpoint"""
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": temperature,
        "stream": True,
        "options": {
            "num_predict": -1,
            "num_ctx": 8192,
        },
        # Enable logprobs (Ollama v0.12.11+)
        "logprobs": True,
        "top_logprobs": 5  # Get top 5 alternative tokens
    }
    
    try:
        response = requests.post(OLLAMA_API, json=payload, stream=True, timeout=None)
        response.raise_for_status()
        
        full_text = ""
        token_logprobs = []
        
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_response = json.loads(line)
                    
                    if "error" in json_response:
                        print(f"\n\033[91mOllama Error: {json_response['error']}\033[0m")
                        break
                    
                    # Extract response text
                    chunk = json_response.get("response", "")
                    
                    # Extract logprobs if present
                    logprobs_data = json_response.get("logprobs", None)
                    
                    if logprobs_data:
                        # Process each token's logprobs
                        for logprob_item in logprobs_data:
                            token = logprob_item.get("token", "")
                            logprob = logprob_item.get("logprob", 0.0)
                            prob = np.exp(logprob)
                            top_lp = logprob_item.get("top_logprobs", [])
                            
                            # Store token info
                            token_info = TokenLogProb(
                                token=token,
                                logprob=logprob,
                                prob=prob,
                                top_logprobs=top_lp
                            )
                            token_logprobs.append(token_info)
                            
                            # Calculate entropy from top_logprobs
                            if top_lp:
                                entropy = calculate_entropy(top_lp)
                            else:
                                entropy = 0.0
                            
                            # Color code the token based on entropy
                            color = get_color_for_entropy(entropy)
                            print(f"{color}{token}\033[0m", end="", flush=True)
                    elif chunk:
                        # No logprobs available, print normally
                        print(chunk, end="", flush=True)
                    
                    full_text += chunk
                    
                    if json_response.get("done", False):
                        break
                        
                except json.JSONDecodeError:
                    continue
        
        print("\n")
        return full_text, token_logprobs

    except Exception as e:
        print(f"\nError generating CoT: {e}")
        return "", []
    
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

def compute_behavioral_metrics(puzzle, cot: str, token_logprobs: List[TokenLogProb]) -> BehavioralMetrics:
    # Calculate entropy metrics
    avg_entropy = 0.0
    high_entropy_tokens = 0
    
    if token_logprobs:
        entropies = []
        for tlp in token_logprobs:
            if tlp.top_logprobs:
                entropy = calculate_entropy(tlp.top_logprobs)
                entropies.append(entropy)
                if entropy > 2.0:  # Threshold for "high uncertainty"
                    high_entropy_tokens += 1
        
        if entropies:
            avg_entropy = np.mean(entropies)
    
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
        avg_entropy=avg_entropy,
        high_entropy_tokens=high_entropy_tokens
    )

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
            
            rating_str = f"(Rating: {puzzle['rating']})" if puzzle['rating'] > 0 else "(Broken/Empty)"
            print(f"\n[{puzzle_count}] ID: {puzzle['id']} {rating_str}")
            print(f"    (Hidden from model) Theme: {puzzle['description'][:70]}...")
            
            prompt = create_puzzle_prompt(puzzle['fen'])
            
            print("\n" + "="*20 + " PROMPT " + "="*20)
            print(prompt)
            print("="*48 + "\n")
            print("-" * 20 + " STREAM START " + "-" * 20)
            
            start_time = time.time()
            cot, token_logprobs = generate_cot(prompt)
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
                rating=puzzle['rating'],
                token_logprobs=token_logprobs
            )
            all_responses.append(response)
            
            metrics = compute_behavioral_metrics(puzzle, cot, token_logprobs)
            all_metrics.append(metrics)
            
            print(f"  CoT length: {metrics.cot_length_tokens} tokens")
            print(f"  Revisions: {metrics.revision_count}, Hedging: {metrics.hedging_count}")
            print(f"  Loop score: {metrics.loop_score:.2f}")
            print(f"  Avg entropy: {metrics.avg_entropy:.3f}, High entropy tokens: {metrics.high_entropy_tokens}")
            
            time.sleep(1.0) 
    
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Save responses with logprobs
    responses_file = OUTPUT_DIR / "cot_responses_with_logprobs.jsonl"
    with open(responses_file, 'w') as f:
        for resp in all_responses:
            # Convert to dict and serialize token_logprobs
            resp_dict = asdict(resp)
            f.write(json.dumps(resp_dict) + '\n')
    print(f"✓ Saved {len(all_responses)} responses with logprobs to {responses_file}")
    
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
        print(f"  Avg entropy: {np.mean([m.avg_entropy for m in diff_metrics]):.3f}")
        print(f"  Avg high entropy tokens: {np.mean([m.high_entropy_tokens for m in diff_metrics]):.1f}")
    
    return all_responses, all_metrics

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ANXIETY MEDS FOR REASONING MODELS - WITH LOGPROBS")
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