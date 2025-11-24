import torch
import requests
import re
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformer_lens import HookedTransformer

# CONFIG
OLLAMA_MODEL = "qwen3:14b"  # Your big thinker
MECH_MODEL = "Qwen/Qwen2.5-0.5B-Instruct" # Small model for fast brain scanning on Mac CPU


def get_ollama_cot(prompt):
    print(f"🤖 Generating CoT with {OLLAMA_MODEL}...")
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.6, "num_predict": 500}, # Cap length for speed
        "stream": False
    })
    return response.json()["message"]["content"]

def run_mac_neuro_scan(prompt, cot_text):
    print(f"🧠 Loading {MECH_MODEL} for brain scan...")
    # Force CPU for stability on Mac, or try "mps" if you feel lucky
    device = "cpu" 
    
    model = HookedTransformer.from_pretrained(
        MECH_MODEL, 
        device=device,
        dtype="float32" # MPS/CPU likes float32 better sometimes
    )
    
    print("   Tracing entropy...")
    # Combine prompt and just the first 100 tokens of CoT to see the "start" of the struggle
    # We truncate to keep it fast
    full_text = prompt + "\n" + " ".join(cot_text.split()[:100])
    
    input_ids = model.to_tokens(full_text)
    
    with torch.no_grad():
        logits = model(input_ids)
        
    # Calculate Entropy
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)[0]
    
    tokens = model.to_str_tokens(input_ids)[0]
    
    # Return just the CoT part (remove prompt)
    prompt_len = len(model.to_tokens(prompt)[0])
    
    return tokens[prompt_len:], entropy[prompt_len-1:-1].numpy()

def main():
    # 1. Define One Easy, One Impossible Case
    easy_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1" # Standard
    hard_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1BROKEN" # Broken
    
    base_prompt = "You are a chess engine. FEN: {fen}. Task: Analyze this position."
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Easy (Valid FEN)", "Impossible (Broken FEN)"))
    
    # RUN EASY
    print("\n--- CASE 1: EASY ---")
    prompt = base_prompt.format(fen=easy_fen)
    cot = get_ollama_cot(prompt)
    tokens, entropy = run_mac_neuro_scan(prompt, cot)
    
    fig.add_trace(go.Scatter(y=entropy, text=tokens, mode='lines+markers', name='Easy', line=dict(color='green')), row=1, col=1)
    
    # RUN IMPOSSIBLE
    print("\n--- CASE 2: IMPOSSIBLE ---")
    prompt = base_prompt.format(fen=hard_fen)
    cot = get_ollama_cot(prompt)
    tokens, entropy = run_mac_neuro_scan(prompt, cot)
    
    fig.add_trace(go.Scatter(y=entropy, text=tokens, mode='lines+markers', name='Impossible', line=dict(color='red')), row=2, col=1)
    
    fig.update_layout(height=800, hovermode="x")
    fig.write_html("results/mac_struggle_scan.html")
    print("\n✅ Saved scan to results/mac_struggle_scan.html")

if __name__ == "__main__":
    main()