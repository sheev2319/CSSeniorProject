from stockfish import Stockfish
import numpy as np
import time

NUM_OPTIONS = 5
BETA = 0.5

def func(x):
	if not np.isinf(x):
		return np.exp(x)
	elif x < 0:
		return 0
	else:
		return np.inf

vec_exp = np.vectorize(func)

class StockfishWrapper:
	def __init__(self) -> None:
		# path = "./stockfish/src/stockfish"
		self.stockfish = Stockfish(parameters={"Hash": 2048, "UCI_Chess960": True, "Minimum Thinking Time": 0,
								"Move Overhead": 0, "Slow Mover": 0})
		self.stockfish.set_depth(10)

	def get_turn(self):
		"""Returns 1 if white's turn, -1 if black's turn"""
		fen = self.stockfish.get_fen_position()
		parts = fen.split()
		if parts[1] == "w": # white to move
			return 1
		elif parts[1] == "b": # black to move
			return -1
		else:
			raise Exception

	def evaluate_terminal(self, move_number=10):
		"""Checks if the position is terminal. If the position is terminal, returns
		(1, 'terminal') for white win, (-1, 'terminal') for black win, and (0,'terminal') for draw.
		If position is not terminal, returns (centipawn advantage, None)"""
		turn = self.get_turn()
		evaluation = self.stockfish.get_evaluation()
		if evaluation['type'] == 'mate':
			# game is over
			if evaluation['value'] > 0:
				# player to move has won the game
				return (turn, "terminal")
			else:
				# other player has won the game
				return (-turn, "terminal")
		elif self.stockfish.get_top_moves(1) == []:
			# position is a draw
			return (0, "terminal")
		else:
			# game is not necessarily over
			if move_number >= 120 and abs(evaluation['value'])<=30:
				return (0, "terminal")
			elif evaluation['value'] >= 400:
				return (1, "terminal")
			elif evaluation['value'] <= -400:
				return (-1, "terminal")
			else: return (evaluation['value'],None)
		
	def sample_move_get_entropy(self):
		# assumes there is at least one legal move available
		# use evaluate_terminal() to check before calling this !!
		
		top_moves = self.stockfish.get_top_moves(NUM_OPTIONS)
		if len(top_moves) == 0:
			return (None,1,0)
		turn = self.get_turn() # 1 for White, -1 for Black
		moves = [item['Move'] for item in top_moves] # extract moves
		evaluations = np.array([turn * item['Centipawn'] / 10 if item['Centipawn'] is not None 
									else np.inf * turn * (1 if item['Mate'] > 0 else -1) for item in top_moves])  
		weights = vec_exp(evaluations * BETA) # softmax exponentiation
		if not np.isinf(sum(weights)) and sum(weights)!=0:
			# normal case
			weights = weights/sum(weights) # softmax normalization
		else: 
			# either all moves result in a loss or there's a winning move
			# either way, play the first move with probability 1
			weights = np.zeros(len(weights))
			weights[0]=1      
		selected_index = np.random.choice(range(len(moves)), p=weights)
		taken_move = moves[selected_index]
		move_probability = weights[selected_index]  
		# compute entropy of state
		entropy = 0
		for i in range(len(moves)):
			if weights[i] > 0:
				entropy -= weights[i]*np.log(weights[i])  
		#print(weights) 
		return (taken_move, move_probability, entropy)


	def run_simulation(self,num_iters,fisher_number,fen_position):
		# average_time = 0
		start_time = time.time()
		
		data = []
		iteration_counter = 0
		num_errors = 0
		while iteration_counter < num_iters:
			try:
				self.stockfish.set_fen_position(fen_position)
				# start_time = time.time()
				i = 0

				entropy_current = []
				transition_probability_current = []
				taken_move_current = []
				evaluation_current = []
				while(True):
					#if i % 2 == 0 and i <= 20:
						#print(self.stockfish.get_board_visual())
					# time_1 = time.time()
					centipawns, outcome = self.evaluate_terminal(i)
					if outcome == "terminal":
						# game is over
						### If you come here, centipawns is the game outcome
						game_outcome = centipawns
						break 
					if i <= 40:
						taken_move, move_probability, entropy = self.sample_move_get_entropy()
						taken_move_current.append(taken_move)
						transition_probability_current.append(move_probability)
						entropy_current.append(entropy)
					elif i <= 80:
						taken_move = self.stockfish.get_best_move_time(40)
					else:
						taken_move = self.stockfish.get_best_move_time(20)
					self.stockfish.make_moves_from_current_position([taken_move])
					evaluation_current.append(centipawns)
					# time_2 = time.time()
					#print(time_2-time_1)
					i += 1
				
				data.append([fisher_number,tuple(entropy_current),
					tuple(transition_probability_current),tuple(taken_move_current),
					tuple(evaluation_current),game_outcome])
				iteration_counter +=1
			except:
				num_errors +=1
				
		end_time = time.time()
		print(f"Time taken: {end_time-start_time}")
		return data,num_errors

"""
iters=1
wrapper = StockfishWrapper()
data, num_errors = wrapper.run_simulation(1, 0, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
"""