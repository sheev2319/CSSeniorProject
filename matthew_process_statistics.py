from itertools import islice
import numpy as np
import math
from data_functions import get_data,NUM_LINES


def uncertainty(evaluations):
	G = len(evaluations)
	S = 100
	uncertainty = 0
	for s in range(S):
		t = s/S #***** not s/(S-1) because of indexing
		sm = 0
		for eval_co in evaluations:
			ind = int(t*len(eval_co))
			sm += eval_co[ind]
			
		sm = sm / G
		uncertainty += min(1,t-sm)
	return uncertainty / S
			
		
def killer_moves(evaluations):
	killer = 0
	for eval_co in evaluations:
		move_diff = [eval_co[i+1] + eval_co[i] for i in range(len(eval_co)-2)]
		killer += max(move_diff)
	return killer / len(evaluations)

def permanence(evaluations):
	perm = 0
	for eval_co in evaluations:
		inner = 0
		num_triples = len(eval_co)-2
		for i in range(num_triples):
			inner += eval_co[i+2]+eval_co[i+1]+eval_co[i+1]+eval_co[i]
		perm += 1- inner/num_triples
	return perm/len(evaluations)

def get_leads(eval_co):
	leader = []
	is_start = True
	for i in range(len(eval_co)):
		eval = math.pow(-1,i)*eval_co[i]
		if eval > 0:
			is_start = False
			leader.append(1)
		elif eval < 0:
			is_start = False
			leader.append(-1)
		elif is_start:
			is_start = False
			leader.append(1)
		else:
			leader.append(leader[-1])
	return leader

	

def lead_change(evaluations):
	lead = 0
	for eval_co in evaluations:
		leader = get_leads(eval_co)
		lead += np.mean([leader[i] != leader[i+1] for i in range(len(leader)-1)])
	return lead / len(evaluations)


def completion(outcomes):
	return np.mean([outcome != 0 for outcome in outcomes])

def duration(evaluations):
	game_lengths = [len(eval_co) for eval_co in evaluations]
	# return np.std(game_lengths)/np.mean(game_lengths)
	m_pref = np.mean(game_lengths)
	return np.mean([abs(m_pref-length)/m_pref for length in game_lengths])



def main():
	# inds = [(120*i,120*(i+1)-1) for i in range(8)]
	
	
	stats = [[] for _ in range(6)]
	filename = f"filtered_data.txt"
	# print(read_last_line(filename))
	counter = 0
	with open(filename, 'r') as file:
		for _ in range(960): # should be NUM_POSITIONS
			evaluations,outcomes = get_data(list(islice(file, NUM_LINES)),counter)
			counter += NUM_LINES
			stats[0].append(uncertainty(evaluations))
			stats[1].append(killer_moves(evaluations))
			stats[2].append(permanence(evaluations))
			stats[3].append(lead_change(evaluations))
			stats[4].append(completion(outcomes))
			stats[5].append(duration(evaluations))


	
	ordered_weights = np.array([0.2023,0.3585,0.1027,-0.2769,0.0788,-0.0907])
	ordered_weights = ordered_weights/np.sum(ordered_weights)
	stats = np.array(stats)
	stats = stats.T
	final_scores = np.matmul(stats,ordered_weights)

	data = np.c_[stats, final_scores]
	print(data.shape)
	np.savez("matthew_final.npz",arr1 = data)
	# # with open("final_output.txt",'w') as file:
	# lines = [f'{" ".join([str(j) for j in stats[:,i]])} {final_scores[i]}\n' for i in range(len(final_scores))]

	# with open("final_results_flipped.txt",'w') as file:
	# 	file.writelines(lines)


# a= np.array([[j*i+i+j for i in range(3)]for j in range(2)])
# weights=np.ones(3)
# print(a)
# print(a.shape)
# b = np.matmul(weights,a)
# print(b)
# print(b.shape)
# c = np.c_[a, b]
# print(c)
# print(c.shape)



main()

