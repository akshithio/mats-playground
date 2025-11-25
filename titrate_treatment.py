import torch
import numpy as np
from transformer_lens import HookedTransformer

# CONFIG
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER_TO_STEER = 12

def get_steering_vector(model, layer):
    print("🧪 Synthesizing Anxiety Meds...")
    
    # 1. Easy (Confidence)
    easy_prompts = [
        "FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 Task: Analyze.",
        "FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 Task: Analyze."
    ]
    tokens_easy = model.to_tokens(easy_prompts)
    _, cache_easy = model.run_with_cache(tokens_easy)
    act_easy = cache_easy[f"blocks.{layer}.hook_resid_post"][:, -1, :].mean(dim=0)

    # 2. Hard (Anxiety)
    hard_prompts = [
        "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1BROKEN Task: Analyze.",
        "FEN: 8/8/8/8/8/8/8/8 w - - 0 1 Task: Analyze."
    ]
    tokens_hard = model.to_tokens(hard_prompts)
    _, cache_hard = model.run_with_cache(tokens_hard)
    act_hard = cache_hard[f"blocks.{layer}.hook_resid_post"][:, -1, :].mean(dim=0)

    # Direction: Toward Confidence, Away from Anxiety
    return act_easy - act_hard

def main():
    print(f"🧠 Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu", dtype="float32")
    
    steering_vec = get_steering_vector(model, LAYER_TO_STEER)
    
    # The Patient: A broken puzzle that normally confuses the model
    patient_prompt = "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1BROKEN Task: Analyze."
    
    print(f"\n📝 PROMPT: {patient_prompt}")

    # --- CONTROL ---
    print(f"\n--- DOSE: 0.0 (Placebo) ---")
    output = model.generate(patient_prompt, max_new_tokens=40, verbose=False)
    # Strip prompt for cleaner reading
    print(output.replace(patient_prompt, "").strip())

    # --- EXPERIMENT SWEEP ---
    # We try: 
    #  0.5 (Low Dose)
    #  1.0 (Standard Dose)
    # -1.0 (Anxiety Inducing / Negative Control)
    doses = [0.5, 1.0, -1.0]
    
    for strength in doses:
        label = "CONFIDENCE BOOST" if strength > 0 else "ANXIETY ATTACK"
        print(f"\n--- DOSE: {strength} ({label}) ---")
        
        def hook_fn(resid, hook):
            # resid shape: [batch, pos, d_model]
            return resid + (steering_vec * strength)
            
        with model.hooks(fwd_hooks=[(f"blocks.{LAYER_TO_STEER}.hook_resid_post", hook_fn)]):
            output = model.generate(patient_prompt, max_new_tokens=40, verbose=False)
            
        print(output.replace(patient_prompt, "").strip())

if __name__ == "__main__":
    main()