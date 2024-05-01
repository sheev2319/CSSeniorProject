import time
import numpy as np
import concurrent.futures
from stockfishWrapper import StockfishWrapper

NUM_ITERS = 750
N_PROC = 8

def batch_simulations(inputs):
	wrapper = StockfishWrapper()
	start = time.perf_counter()
	'''runs a series of simulations for each position input
	inputs: list of tuples where tuples are of format
	(960 position number, fen_position)
	saves the results to a file with name based on the range of 960 positions
	'''
	cols = ["960 Position Number","Entropy","Transition Probabilities","Moves Taken",
		 "Evaluations","Game Outcome"]
	
	filename = f"data/game_data_{inputs[0][0]}-{inputs[-1][0]}.txt"
	logname = f"logs/log_{inputs[0][0]}-{inputs[-1][0]}.txt"

	cur = next_position(filename,inputs)
	num_errors = 0
	for i in range(cur,len(inputs)):
		with open(logname, 'a') as file:
			file.write(f'{i}\n')
		data,cur_errors = wrapper.run_simulation(NUM_ITERS,inputs[i][0],inputs[i][1])
		num_errors += cur_errors
		lines = ["_".join([str(j) for j in row])+"\n" for row in data]
		with open(filename, 'a') as file:
			file.writelines(lines)

	'''inputs[next_position(filename)%120]'''
	end = time.perf_counter()
	return ((end-start)/(len(inputs)*NUM_ITERS),num_errors)

def next_position(filename,inputs):
	'''Returns the next 960 position to process'''
	mx = -1
	with open(filename,"r") as file:
		for line in file:
			game_number = int(line.split("_")[0])
			if mx < game_number:
				mx = game_number
	next = mx +1
	for i in range(len(inputs)):
		if inputs[i][0] == next:
			return i
	return len(inputs)

def read_last_line(filename):
	try:
		with open(filename, 'rb') as f:
			f.seek(0, 2)  # Move the cursor to the end of the file
			end_byte = f.tell()
			while f.tell() > 0:
				f.seek(-2, 1)  # Move the cursor backwards by two bytes
				if f.read(1) == b'\n':  # Check if it's the beginning of a line
					return f.readline().decode()  # Read and return the last line
			f.seek(0)  # If the file doesn't have any newline, go to the start
			return f.readline().decode()  # Read the first line, since it's also the last in this case
	except:
		return ""

def get_all_fen_positions():
	positions = np.empty(960, dtype=object)
	with open("fen_positions.txt", 'r') as file:
		i = 0
		for line in file:

			positions[i]=(i,line[:-1])
			i+=1
	return positions

def main():
	chunks = np.array_split(get_all_fen_positions(),N_PROC)
	with concurrent.futures.ProcessPoolExecutor() as executor:
		executor.map(batch_simulations,chunks)

# inputs = [(3,"bnqnrkrb/pppppppp/8/8/8/8/PPPPPPPP/BNQNRKRB w KQkq - 0 1"),(1,"bnqnrkrb/pppppppp/8/8/8/8/PPPPPPPP/BNQNRKRB w KQkq - 0 1")]
# batch_simulations(inputs)
# get_all_fen_positions()
		
main()