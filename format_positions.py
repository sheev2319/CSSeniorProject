

"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
fens = []
with open('raw_positions.txt', 'r') as file:
	for line in file:
		
		parts = line.split()
		white = parts[1]
		black = parts[2]
		fens.append(f'{black}/pppppppp/8/8/8/8/PPPPPPPP/{white} w KQkq - 0 1')

print(len(fens))

with open('fen_positions.txt', 'w') as file:
	for fen in fens:
		file.write(fen + '\n')
