import torch
import numpy as np
import time
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich import box
from rich.align import Align

# ================= CONFIGURATION =================
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Global containers
layer_activations = []
mid_layer_hidden_state = None 
MID_LAYER_IDX = 14 

def make_layer_hook(layer_idx):
    def hook(module, input, output):
        global mid_layer_hidden_state
        
        if isinstance(output, tuple):
            hidden_state = output[0]
        else:
            hidden_state = output
            
        # Move to CPU immediately to avoid MPS synchronization bugs
        # We only keep the "Spy Vector" on the GPU/MPS for the next calculation
        if len(hidden_state.shape) == 3:
            if layer_idx == MID_LAYER_IDX:
                mid_layer_hidden_state = hidden_state[0, -1, :].detach()
            val = hidden_state[0, -1, :].float().norm().item()
        elif len(hidden_state.shape) == 2:
            if layer_idx == MID_LAYER_IDX:
                mid_layer_hidden_state = hidden_state[-1, :].detach()
            val = hidden_state[-1, :].float().norm().item()
        else:
            val = 0.0
            
        if layer_idx < len(layer_activations):
            layer_activations[layer_idx] = val
    return hook

# ================= VISUALIZATION =================
def make_layout():
    layout = Layout()
    layout.split_row(
        Layout(name="left", ratio=1), 
        Layout(name="center", ratio=1), 
        Layout(name="right", ratio=1), 
    )
    layout["left"].split_column(
        Layout(name="header", size=3),
        Layout(name="stream", ratio=1),
        Layout(name="stats", size=4),
    )
    layout["center"].split_column(
        Layout(name="final_probs", ratio=1),
        Layout(name="subconscious", ratio=1),
    )
    layout["right"].split_column(
        Layout(name="entropy", size=3),
        Layout(name="spine", ratio=1),
    )
    return layout

def generate_header():
    return Panel(
        Text(f"🧠 NEUROSCOPE: ROBUST MODE", justify="center", style="bold magenta"),
        style="white on black"
    )

def generate_stats_panel(tps, token_count):
    return Panel(
        Text(f"Speed: {tps:.1f} tok/s | Count: {token_count}", justify="center", style="bold cyan"),
        title="Inference Stats", border_style="cyan"
    )

def generate_entropy_panel(entropy):
    if entropy is None or np.isnan(entropy): entropy = 0.0
    color = "green" if entropy < 0.6 else "yellow" if entropy < 2.5 else "red"
    return Panel(
        Align.center(Text(f"Uncertainty: {entropy:.2f}", style=f"bold {color}")),
        border_style=color
    )

def generate_spine_panel(num_layers):
    max_val = 300.0 
    text = Text()
    text.append("Output (Final)\n", style="dim white")
    
    for i in range(num_layers - 1, -1, -1):
        val = layer_activations[i] if i < len(layer_activations) else 0.0
        if np.isnan(val): val = 0.0
        percent = min(val / max_val, 1.0)
        bars = int(percent * 20)
        
        if i == MID_LAYER_IDX:
            style = "bold yellow"
            marker = "👁️ "
        else:
            style = "blue" if i < 5 else "cyan" if i < 20 else "magenta"
            marker = "L "
            
        bar_visual = "█" * bars
        text.append(f"{marker}{i:02d} │ {bar_visual} \n", style=style)
        
    text.append("Input (Raw)\n", style="dim white")
    return Panel(text, title="Neural Activity", border_style="white")

def generate_probs_table(top_tokens, top_probs, title, color_style="green"):
    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    for token, prob in zip(top_tokens, top_probs):
        if np.isnan(prob): prob = 0.0
        c = color_style if prob > 0.5 else "white"
        table.add_row(token, f"[{c}]{prob:.1%}[/]")
    return Panel(table, title=title, border_style=color_style)

# ================= EXECUTION =================
def run():
    console = Console()
    console.print(f"[bold cyan]Loading {MODEL_ID}...[/bold cyan]")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16, 
            device_map="mps",
            attn_implementation="eager"
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return

    num_layers = len(model.model.layers)
    global layer_activations
    layer_activations = [0.0] * num_layers
    
    for i in range(num_layers):
        model.model.layers[i].register_forward_hook(make_layer_hook(i))

    puzzle_fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 5"
    task_prompt = f"Analyze this chess position (FEN: {puzzle_fen}). Find the best move."
    
    system_prompt = """You are a chess expert. 
    Format your response EXACTLY like this:
    <think>
    [Write your step-by-step analysis here]
    </think>
    [Write your final answer and move here]
    """
    
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": task_prompt}
    ]
    
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to("mps")
    
    # KV Cache variables
    input_ids = inputs.input_ids
    past_key_values = None
    
    full_text_display = Text()
    full_text_display.append("PROMPT:\n", style="bold magenta")
    full_text_display.append(f"{task_prompt}\n", style="magenta")
    full_text_display.append("─" * 30 + "\n", style="dim white")
    
    temperature = 0.7
    layout = make_layout()
    layout["left"]["header"].update(generate_header())

    start_time = time.time()
    token_count = 0
    is_thinking = False 

    # === CRASH PROTECTION: TRY/EXCEPT BLOCK ===
    try:
        with Live(layout, refresh_per_second=10, screen=True) as live:
            for _ in range(800): 
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids, 
                        past_key_values=past_key_values, 
                        use_cache=True
                    )
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[0, -1, :].float() # float32 for safety
                    
                    # --- Subconscious Spy (Protected) ---
                    try:
                        if mid_layer_hidden_state is not None:
                            mid_vec = mid_layer_hidden_state.to("mps").to(torch.bfloat16)
                            mid_vec = model.model.norm(mid_vec)
                            mid_logits = model.lm_head(mid_vec).float()
                            mid_probs = torch.softmax(mid_logits, dim=0)
                            mid_top_probs, mid_top_indices = torch.topk(mid_probs, 5)
                        else:
                            raise ValueError("No hidden state")
                    except Exception:
                        # If spy fails, just return empty (don't crash the main loop)
                        mid_top_probs, mid_top_indices = torch.tensor([]), torch.tensor([])

                    # --- Main Generation ---
                    probs_func = torch.nn.Softmax(dim=0)
                    probs_for_entropy = probs_func(next_token_logits)
                    entropy = -torch.sum(probs_for_entropy * torch.log(probs_for_entropy + 1e-9)).item()
                    
                    scaled_logits = next_token_logits / temperature
                    probs = probs_func(scaled_logits)
                    
                    # Safety against NaNs
                    if torch.isnan(probs).any(): 
                        probs = torch.ones_like(probs) / len(probs)
                    
                    top_probs, top_indices = torch.topk(probs, 5)
                    next_token_id = torch.multinomial(probs, num_samples=1)

                new_word = tokenizer.decode(next_token_id[0])
                input_ids = next_token_id.unsqueeze(0) 
                token_count += 1
                
                # --- Text Parsing ---
                if "<think>" in new_word:
                    is_thinking = True
                    full_text_display.append("\n🧠 THINKING:\n", style="bold cyan underline")
                    new_word = new_word.replace("<think>", "")
                    style = "italic cyan"
                elif "</think>" in new_word:
                    is_thinking = False
                    new_word = new_word.replace("</think>", "")
                    full_text_display.append(new_word, style="italic cyan")
                    full_text_display.append("\n\n🎯 ANSWER:\n", style="bold green underline")
                    new_word = "" 
                    style = "bold white"
                else:
                    style = "italic cyan" if is_thinking else "bold white"
                
                full_text_display.append(new_word, style=style)

                # Buffer Management: Keep text object from growing infinitely
                # If text is massive, we might want to trim (optional, but good for stability)
                # For now, we rely on Rich handling it.

                # --- Update UI ---
                top_str = [tokenizer.decode([idx]) for idx in top_indices]
                
                if len(mid_top_indices) > 0:
                    mid_str = [tokenizer.decode([idx]) for idx in mid_top_indices]
                    mid_vals = mid_top_probs.tolist()
                else:
                    mid_str, mid_vals = ["-"], [0.0]

                elapsed = time.time() - start_time
                tps = token_count / (elapsed + 1e-9)
                
                layout["left"]["stream"].update(Panel(full_text_display, title="Stream"))
                layout["left"]["stats"].update(generate_stats_panel(tps, token_count))
                layout["center"]["final_probs"].update(generate_probs_table(top_str, top_probs.tolist(), "Final (L28)", "green"))
                layout["center"]["subconscious"].update(generate_probs_table(mid_str, mid_vals, "Subconscious (L14)", "yellow"))
                layout["right"]["entropy"].update(generate_entropy_panel(entropy))
                layout["right"]["spine"].update(generate_spine_panel(num_layers))
                
                if next_token_id.item() in [tokenizer.eos_token_id, 151645, 151643]:
                    break

    except Exception as e:
        # If crash happens, Live context closes, and we print the error here
        console.print(f"\n[bold red]CRASH DETECTED:[/bold red] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run()