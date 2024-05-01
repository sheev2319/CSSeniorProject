import ast

'''
cols = ["960 Position Number","Entropy","Transition Probabilities","Moves Taken",
		 "Evaluations","Game Outcome"]

indices = {}
for i in range(len(cols)):
	indices[cols[i]] = i
'''

def read_data(line):
	data = line[:-1].split("_")
	position_number = int(data[0])
	entropy = ast.literal_eval(data[1])
	transition_probabilities = ast.literal_eval(data[2])
	moves_taken = ast.literal_eval(data[3])
	evaluations = ast.literal_eval(data[4])
	outcome = int(data[5])
	return [position_number,entropy,transition_probabilities,moves_taken,evaluations,outcome]

def read_game(file_path):
	with open(file_path, 'r') as file:
		for line in file:
			data = read_data(line)
			

read_game("data/game_data.txt")