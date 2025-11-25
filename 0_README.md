Commit 1: Just General Running of Experiments with Claude-Generated Positions (that were inaccurate) to get the system to work. Results showed that the looping rate on basically every puzzle was 1.0 aka this meant that every single puzzle that we tested had a sentence repeat over.

Commit 2: Puzzles chosen from Lichess based on difficulty. For me, it is still running currently but what I notice is a:

- a lack of metacognitional ability (to detect that puzzles are growing in difficulty) to actually solve the puzzles because CoT time taken stays the same as the difficulty of the puzzle grows (around 180 seconds). This could have to do with the CoT token limit (= 4000 right now) that has been set instead. (Update: It doesn't seem like it because CoT length seems to be around 2000 - 3000ish tokens?)

- There is also the possibility because it spends so much more time understanding the notation than trying to solve the puzzle, that to measure difficulty we need to just give it FEN positions instead of puzzles. We start with a simple position that has only two kings on the board and a rook or something and ramp it up to FENs that have more pieces and more possible move orders.

- I am also intrigued about additional questions like how would it start to think differently, if told that a puzzle's rating versus when it isn't and what changes internally then

- Should I also be switching to the 7B version so I can actually do this stuff locally?