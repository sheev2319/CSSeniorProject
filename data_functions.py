import ast
import math
import numpy as np

NUM_LINES = 750
FLIPPED = 0


with open("bad_positions.txt",'r') as file:
	bad_positions = ast.literal_eval(file.read())

# bad_positions = set(bad_positions)

def read_game(line):
	data = line[:-1].split("_")
	position_number = int(data[0])
	# entropy = ast.literal_eval(data[1])
	# transition_probabilities = ast.literal_eval(data[2])
	# moves_taken = ast.literal_eval(data[3])
	evaluations = ast.literal_eval(data[4])
	outcome = int(data[5])
	# return [position_number,entropy,transition_probabilities,moves_taken,evaluations,outcome]
	return [position_number,evaluations,outcome]


def get_data(lines,counter):
	fen_number = None
	evaluations = []
	outcomes = []
	for line in lines:
		if counter in bad_positions:
			counter +=1
			continue
		position_number,game_evaluations,outcome = read_game(line)
		if fen_number == None:
			fen_number = position_number
		if fen_number != None and fen_number != position_number:
			raise ValueError
		outcomes.append(outcome)
		eval_co = list(get_board_evaluations(game_evaluations))
		eval_co.append(outcome)
		eval_co = [math.pow(-1,i+FLIPPED)*eval_co[i] for i in range(len(eval_co))]
		evaluations.append(eval_co)
		counter +=1
	
	return evaluations,outcomes

def board_evaluation_calculator(evaluation):
	#**** win probability for white
	# return 1/(1+math.pow(10,-evaluation/400))

	# differene in board evaluation
	# 1 = white win
	# -1 = black win
	return 2/(1+math.pow(10,-evaluation/400))-1

get_board_evaluations = np.vectorize(board_evaluation_calculator)