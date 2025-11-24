import torch
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformer_lens import HookedTransformer

# CONFIG
OLLAMA_MODEL = "qwen3:14b"
MECH_MODEL = "Qwen/Qwen2.5-0.5B-Instruct" 

def get_ollama_cot(prompt):
    try:
        response = requests.post("http://localhost:11434/api/chat", json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.6, "num_predict": 200}, # Short CoT for visualization
            "stream": False
        })
        return response.json()["message"]["content"]
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""

def run_neuro_scan(prompt, cot_text, model):
    # Combine and tokenize
    full_text = prompt + "\n" + cot_text
    input_ids = model.to_tokens(full_text)
    
    with torch.no_grad():
        logits = model(input_ids)
        
    # Calculate Entropy
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)[0]
    
    tokens = model.to_str_tokens(input_ids)[0]
    
    # SAFER SLICING: Just take the last N tokens corresponding to the CoT
    # We estimate CoT length by tokenizing it separately
    cot_len = len(model.to_tokens(cot_text)[0])
    
    # If CoT is too short, just plot everything so we see SOMETHING
    start_idx = max(0, len(tokens) - cot_len - 1)
    
    return tokens[start_idx:], entropy[start_idx-1:-1].numpy()

def main():
    print(f"🧠 Loading {MECH_MODEL}...")
    model = HookedTransformer.from_pretrained(MECH_MODEL, device="cpu", dtype="float32")

    # 1. EASY CASE
    print("\n--- RUNNING EASY CASE ---")
    easy_prompt = "You are a chess engine. FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1. Analyze."
    easy_cot = get_ollama_cot(easy_prompt)
    print(f"📝 Generated CoT ({len(easy_cot)} chars): {easy_cot[:50]}...")
    
    tok_easy, ent_easy = run_neuro_scan(easy_prompt, easy_cot, model)

    # 2. IMPOSSIBLE CASE
    print("\n--- RUNNING IMPOSSIBLE CASE ---")
    hard_prompt = "You are a chess engine. FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1BROKEN. Analyze."
    hard_cot = get_ollama_cot(hard_prompt)
    print(f"📝 Generated CoT ({len(hard_cot)} chars): {hard_cot[:50]}...")
    
    tok_hard, ent_hard = run_neuro_scan(hard_prompt, hard_cot, model)

    # 3. PLOT
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Easy (Confident)", "Impossible (Struggling)"))
    
    fig.add_trace(go.Scatter(y=ent_easy, text=tok_easy, mode='lines+markers', line=dict(color='green'), name="Easy"), row=1, col=1)
    fig.add_trace(go.Scatter(y=ent_hard, text=tok_hard, mode='lines+markers', line=dict(color='red'), name="Impossible"), row=2, col=1)
    
    fig.update_layout(height=800, title_text="The Shape of Anxiety")
    fig.write_html("results/fixed_scan.html")
    print("\n✅ Saved to results/fixed_scan.html")

if __name__ == "__main__":
    main()