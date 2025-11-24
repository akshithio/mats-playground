import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

# CONFIG
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER_TO_STEER = 12
INJECTION_STRENGTH = 2.0  # How much "drug" to give the model. Try 0.5, 2.0, 5.0

def get_steering_vector(model, layer):
    """
    Calculates the 'Confidence Vector' (Mean(Easy) - Mean(Hard))
    """
    print("🧪 Synthesizing Anxiety Meds (Computing Steering Vector)...")
    
    # 1. Get Easy Activations
    easy_prompts = [
        "FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 Task: Analyze.",
        "FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 Task: Analyze."
    ]
    tokens_easy = model.to_tokens(easy_prompts)
    _, cache_easy = model.run_with_cache(tokens_easy)
    # Get average activation of the last token
    act_easy = cache_easy[f"blocks.{layer}.hook_resid_post"][:, -1, :].mean(dim=0)

    # 2. Get Hard Activations
    hard_prompts = [
        "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1BROKEN Task: Analyze.",
        "FEN: 8/8/8/8/8/8/8/8 w - - 0 1 Task: Analyze."
    ]
    tokens_hard = model.to_tokens(hard_prompts)
    _, cache_hard = model.run_with_cache(tokens_hard)
    act_hard = cache_hard[f"blocks.{layer}.hook_resid_post"][:, -1, :].mean(dim=0)

    # 3. Compute Direction: "Away from Hard, Toward Easy"
    steering_vec = act_easy - act_hard
    return steering_vec

def main():
    print(f"🧠 Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu", dtype="float32")
    
    # 1. Create the 'Drug'
    steering_vec = get_steering_vector(model, LAYER_TO_STEER)
    
    # 2. Select a Patient (A hard/broken puzzle)
    # We use a broken FEN that usually causes loop/anxiety
    patient_prompt = "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1BROKEN Task: Analyze."
    
    print("\n--- CONTROL GROUP (No Intervention) ---")
    output_base = model.generate(patient_prompt, max_new_tokens=60, verbose=False)
    print(f"Patient: {output_base}")

    # 3. Define the Injection Hook
    def anxiety_med_hook(resid_post, hook):
        # resid_post shape: [batch, pos, d_model]
        # We add the vector to EVERY token position to maintain confidence throughout
        
        # We need to reshape steering_vec to match batch/pos
        # It broadcasts automatically, but let's be safe
        return resid_post + (steering_vec * INJECTION_STRENGTH)

    # 4. Apply Treatment
    print(f"\n--- EXPERIMENTAL GROUP (Strength {INJECTION_STRENGTH}x) ---")
    
    # We use run_with_hooks to generate text while modifying activations live
    with model.hooks(fwd_hooks=[(f"blocks.{LAYER_TO_STEER}.hook_resid_post", anxiety_med_hook)]):
        output_steered = model.generate(patient_prompt, max_new_tokens=60, verbose=False)
        
    print(f"Patient: {output_steered}")
    
    # 5. Analysis
    print("\n--- CLINICAL NOTES ---")
    if "wait" in output_base.lower() and "wait" not in output_steered.lower():
        print("✅ SUCCESS: Intervention removed hesitation.")
    elif len(output_steered) < len(output_base):
        print("ℹ️ NOTE: Intervention changed output length.")
    else:
        print("❓ OBSERVATION: Check the text manually above.")

if __name__ == "__main__":
    main()