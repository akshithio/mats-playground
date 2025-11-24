import csv
import random
from pathlib import Path

def get_puzzles(csv_path="lichess_db_puzzle.csv", samples_per_category=5):
    path = Path(csv_path)
    if not path.exists():
        print(f"Warning: {csv_path} not found. Returning empty dictionary.")
        return {}

    categories = {
        "easy": (600, 1000),       # Beginner
        "medium": (1400, 1800),    # Intermediate
        "hard": (2000, 2400),      # Advanced
        "impossible": (2600, 3500) # Grandmaster / Engine lines
    }

    candidates = {k: [] for k in categories}

    print(f"Loading puzzles from {csv_path}...")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rating = int(row['Rating'])
                    
                    # Sort into buckets
                    for cat, (low, high) in categories.items():
                        if low <= rating <= high:
                            candidates[cat].append(row)
                            break
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}

    final_puzzles = {}

    for cat in categories:
        available = candidates[cat]
        if not available:
            print(f"Warning: No puzzles found for category {cat}")
            final_puzzles[cat] = []
            continue
            
        selection = random.sample(available, min(len(available), samples_per_category))
        
        formatted_list = []
        for p in selection:
            # Parse FEN to see who moves
            fen = p['FEN']
            turn = "White" if " w " in fen else "Black"
            
            themes = p['Themes'].replace(" ", ", ")
            
            formatted_list.append({
                "id": p['PuzzleId'],
                "fen": fen,
                "description": f"{turn} to move. Themes: {themes}.",
                "solution": p['Moves'].split()[0] if p['Moves'] else "Unknown",
                "difficulty": cat,
                "rating": int(p['Rating'])
            })
        
        final_puzzles[cat] = formatted_list

    # MANUALLY ADD: The "Broken" puzzles for the 'impossible' category
    # These are crucial for testing parser anxiety (errors vs logic).
    final_puzzles["impossible"].append({
        "id": "BROKEN_FEN",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1BROKEN",
        "description": "Malformed FEN - parsing should fail. Test error handling.",
        "solution": "N/A",
        "difficulty": "impossible",
        "rating": 0
    })
    
    final_puzzles["impossible"].append({
        "id": "EMPTY_BOARD",
        "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
        "description": "Empty board - no pieces to move. Test hallucination.",
        "solution": "N/A",
        "difficulty": "impossible",
        "rating": 0
    })

    print(f"Successfully loaded {sum(len(v) for v in final_puzzles.values())} puzzles.")
    return final_puzzles

if __name__ == "__main__":
    puzzles = get_puzzles()
    print(json.dumps(puzzles, indent=2))