import torch
import numpy as np
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd

# CONFIG
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER_TO_PROBE = 12  # Middle layers usually hold the "reasoning" content

def get_activations(model, prompts):
    """
    Runs the model and extracts the residual stream activations 
    from the very last token of the prompt (the "setup" state).
    """
    activations = []
    
    for prompt in prompts:
        # We only care about the state RIGHT BEFORE it starts generating the CoT
        # because we want to know if the model "smells" the difficulty immediately.
        tokens = model.to_tokens(prompt)
        
        with torch.no_grad():
            # run_with_cache returns (logits, cache)
            _, cache = model.run_with_cache(tokens)
        
        # Extract residual stream from the specified layer
        # Shape: [batch, pos, d_model] -> We take the last token position [-1]
        resid = cache[f"blocks.{LAYER_TO_PROBE}.hook_resid_post"][0, -1, :].cpu().numpy()
        activations.append(resid)
        
    return np.array(activations)

def main():
    print(f"🧠 Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu", dtype="float32")
    
    print("📉 Generating Dataset...")
    
    # DATASET: We need more examples to train a classifier.
    # We will generate synthetic variations of your puzzles.
    
    # Class 0: Easy / Valid
    easy_prompts = [
        "FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 Task: Analyze.",
        "FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3 Task: Analyze.",
        "FEN: 8/8/8/4k3/4P3/8/8/4K3 w - - 0 1 Task: Analyze.", # King and Pawn
        "FEN: rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2 Task: Analyze.",
        "FEN: rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2 Task: Analyze.",
        "FEN: r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3 Task: Analyze.",
        "FEN: rnbqk2r/pppp1ppp/5n2/4p3/1b2P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4 Task: Analyze.",
        "FEN: r1bq1rk1/pppp1ppp/2n2n2/4p3/1b2P3/2N2N2/PPPP1PPP/R1BQKB1R w KQ - 5 5 Task: Analyze.",
    ]
    
    # Class 1: Impossible / Broken
    hard_prompts = [
        "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1BROKEN Task: Analyze.",
        "FEN: 8/8/8/8/8/8/8/8 w - - 0 1 Task: Analyze.", # Empty board
        "FEN: rnzqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Task: Analyze.", # 'z' is not a piece
        "FEN: rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Task: Analyze.", # '9' is invalid
        "FEN: THIS IS NOT A CHESS BOARD BUT PLEASE SOLVE IT Task: Analyze.",
        "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 INVALID Task: Analyze.",
        "FEN: 4k3/8/8/8/8/8/8/4K3 w - - 0 1 BOTH KINGS Task: Analyze.", # Just kings, but labeled weirdly
        "FEN: rrrrrrrr/rrrrrrrr/rrrrrrrr/rrrrrrrr/rrrrrrrr/rrrrrrrr/rrrrrrrr/rrrrrrrr w - - 0 1 Task: Analyze." # All rooks
    ]

    print(f"   Extracting activations from Layer {LAYER_TO_PROBE}...")
    X_easy = get_activations(model, easy_prompts)
    X_hard = get_activations(model, hard_prompts)
    
    # Create Labels (0 = Easy, 1 = Hard)
    y_easy = np.zeros(len(X_easy))
    y_hard = np.ones(len(X_hard))
    
    X = np.concatenate([X_easy, X_hard])
    y = np.concatenate([y_easy, y_hard])
    
    # TRAIN PROBE
    print("\n🤖 Training 'Struggle Detector' Probe...")
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X, y)
    
    # EVALUATE (On training data for now, since dataset is tiny)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    
    print(f"   Probe Accuracy: {acc*100:.1f}%")
    
    if acc > 0.8:
        print("✅ SUCCESS: The model DOES represent difficulty internally!")
    else:
        print("❌ FAILURE: The model's internal state is indistinguishable.")

    # VISUALIZE WITH PCA
    print("\n🎨 Creating PCA Visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Label': ['Easy' if label == 0 else 'Impossible' for label in y],
        'Prompt': easy_prompts + hard_prompts
    })
    
    fig = px.scatter(df, x='PC1', y='PC2', color='Label', hover_data=['Prompt'],
                     title=f"Internal State Separation (Layer {LAYER_TO_PROBE})")
    
    fig.write_html("results/probe_visualization.html")
    print("✅ Saved visualization to results/probe_visualization.html")

if __name__ == "__main__":
    main()