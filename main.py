import json
import re
import requests
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:14b"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

CHESS_PUZZLES = {
    "easy": [
        {
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "description": "White to move. Simple back rank mate threat.",
            "solution": "Bxf7+ (removing defender, leading to checkmate)",
            "difficulty": "easy"
        },
        {
            "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "description": "White to move. Simple center control.",
            "solution": "d4 (opening center)",
            "difficulty": "easy"
        },
        {
            "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "description": "White to move. Basic development.",
            "solution": "Nc3 or Bc4 (develop pieces)",
            "difficulty": "easy"
        },
        {
            "fen": "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            "description": "Black to move. Defend against Bishop pin.",
            "solution": "a6 or Nge7 (deal with pin)",
            "difficulty": "easy"
        },
        {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "description": "Black to move. Respond to e4.",
            "solution": "e5 or c5 (standard openings)",
            "difficulty": "easy"
        },
    ],
    
    "medium": [
        {
            "fen": "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
            "description": "White to move. Tactical sequence with multiple pieces involved.",
            "solution": "Ng5 attacking f7, requires 2-3 move calculation",
            "difficulty": "medium"
        },
        {
            "fen": "r2qkb1r/ppp2ppp/2np1n2/4p1B1/2B1P3/2N2N2/PPP2PPP/R2QK2R b KQkq - 0 7",
            "description": "Black to move. Complex pin situation.",
            "solution": "Be7 unpinning, requires understanding pin mechanics",
            "difficulty": "medium"
        },
        {
            "fen": "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
            "description": "White to move. Requires calculating pawn breaks.",
            "solution": "d4 or a3, multi-move planning needed",
            "difficulty": "medium"
        },
        {
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 5",
            "description": "Black to move. Tactical defense required.",
            "solution": "Na5 or d5, requires seeing tactics",
            "difficulty": "medium"
        },
    ],
    
    "hard": [
        {
            "fen": "r1b2rk1/ppp1qppp/2np1n2/2b1p3/2B1P3/2NPBN2/PPP2PPP/R2Q1RK1 w - - 0 10",
            "description": "White to move. Deep tactical position requiring 4+ move calculation.",
            "solution": "Complex tactical sequence with multiple variations",
            "difficulty": "hard"
        },
        {
            "fen": "r1bq1rk1/pp3pbp/2nppnp1/8/2BNP3/2N1BP2/PPPQ2PP/R4RK1 w - - 0 12",
            "description": "White to move. Multiple candidate moves, requires evaluation.",
            "solution": "Strategic planning with multiple good options",
            "difficulty": "hard"
        },
        {
            "fen": "r2q1rk1/ppp2ppp/2n1pn2/3p4/1b1P4/2NBPN2/PPP2PPP/R1BQR1K1 b - - 0 10",
            "description": "Black to move. Complex middlegame position.",
            "solution": "Requires understanding of pawn structures and piece activity",
            "difficulty": "hard"
        },
    ],
    
    "impossible": [
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1BROKEN",
            "description": "Malformed FEN - parsing should fail.",
            "solution": "N/A - intentionally broken",
            "difficulty": "impossible"
        },
        {
            "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
            "description": "Empty board - no pieces to move.",
            "solution": "N/A - illegal position",
            "difficulty": "impossible"
        },
        {
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 99 99",
            "description": "Extremely complex position requiring 10+ move deep calculation.",
            "solution": "Beyond reasonable calculation depth for this model",
            "difficulty": "impossible"
        },
    ]
}

@dataclass
class CoTResponse:
    puzzle_id: str
    fen: str
    difficulty: str
    prompt: str
    cot: str
    timestamp: float
    
@dataclass
class BehavioralMetrics:
    puzzle_id: str
    difficulty: str
    cot_length: int
    cot_length_tokens: int
    revision_count: int
    hedging_count: int
    loop_score: float
    progress_score: float
    convergence: bool
    appears_correct: bool

def generate_cot(prompt: str, temperature: float = 0.7) -> str:
    api_url = OLLAMA_API.replace("/api/generate", "/api/chat")
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "stream": True,
        "options": {
            "num_predict": 4000,
        }
    }
    
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
                        
                        # FIX: Capture the "thinking" field (Where Qwen 3 puts the CoT)
                        if "thinking" in msg:
                            chunk = msg["thinking"]
                            print(chunk, end="", flush=True)
                            full_text += chunk
                        
                        # Capture the standard "content" field (The final answer)
                        if "content" in msg:
                            chunk = msg["content"]
                            print(chunk, end="", flush=True)
                            full_text += chunk
                    
                    if json_response.get("done", False):
                        break
                        
                except json.JSONDecodeError:
                    continue
                    
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

def compute_behavioral_metrics(puzzle_id: str, difficulty: str, cot: str) -> BehavioralMetrics:
    return BehavioralMetrics(
        puzzle_id=puzzle_id,
        difficulty=difficulty,
        cot_length=len(cot),
        cot_length_tokens=len(cot.split()),
        revision_count=count_revisions(cot),
        hedging_count=count_hedging(cot),
        loop_score=calculate_loop_score(cot),
        progress_score=calculate_progress_score(cot),
        convergence=check_convergence(cot),
        appears_correct=False,
    )

def create_puzzle_prompt(fen: str, description: str) -> str:
    return f"""You are analyzing a chess position. Think step by step.

FEN notation: {fen}

Task: {description}

Please:
1. Parse and understand the position from the FEN notation
2. Analyze the position 
3. Find candidate moves
4. Recommend the best move and explain why

Think through this carefully, step by step."""

def run_baseline_characterization():
    print("=" * 80)
    print("PHASE 2: BASELINE CHARACTERIZATION")
    print("=" * 80)
    
    all_responses = []
    all_metrics = []
    
    puzzle_count = 0
    for difficulty, puzzles in CHESS_PUZZLES.items():
        print(f"\n{difficulty.upper()} puzzles:")
        print("-" * 40)
        
        for i, puzzle in enumerate(puzzles):
            puzzle_id = f"{difficulty}_{i}"
            puzzle_count += 1
            
            print(f"\n[{puzzle_count}] Puzzle: {puzzle['description'][:50]}...")
            
            prompt = create_puzzle_prompt(puzzle['fen'], puzzle['description'])
            
            print("  Generating CoT...\n")
            print("-" * 20 + " STREAM START " + "-" * 20)
            
            start_time = time.time()
            cot = generate_cot(prompt)
            elapsed = time.time() - start_time
            
            print("\n" + "-" * 20 + " STREAM END " + "-" * 20)
            print(f"  Time: {elapsed:.1f}s")
            
            response = CoTResponse(
                puzzle_id=puzzle_id,
                fen=puzzle['fen'],
                difficulty=difficulty,
                prompt=prompt,
                cot=cot,
                timestamp=time.time()
            )
            all_responses.append(response)
            
            metrics = compute_behavioral_metrics(puzzle_id, difficulty, cot)
            all_metrics.append(metrics)
            
            print(f"  CoT length: {metrics.cot_length_tokens} tokens")
            print(f"  Revisions: {metrics.revision_count}, Hedging: {metrics.hedging_count}")
            print(f"  Loop score: {metrics.loop_score:.2f}, Progress: {metrics.progress_score:.2f}")
            print(f"  Converged: {metrics.convergence}")
            
            time.sleep(0.5)
    
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
        print(f"  Avg progress: {np.mean([m.progress_score for m in diff_metrics]):.3f}")
        print(f"  Convergence rate: {sum(m.convergence for m in diff_metrics) / len(diff_metrics) * 100:.1f}%")
    
    return all_responses, all_metrics

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ANXIETY MEDS FOR REASONING MODELS")
    print("Phase 1 & 2: Dataset Construction + Baseline Characterization")
    print("=" * 80)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total puzzles: {sum(len(p) for p in CHESS_PUZZLES.values())}")
    
    responses, metrics = run_baseline_characterization()
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)