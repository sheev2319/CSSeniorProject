import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import islice
from scipy.stats import norm,ttest_ind
from data_functions import get_data,NUM_LINES


# data = np.load("matthew_final.npz",allow_pickle=True)
# data = data['arr1']

names=  ["Uncertainty", "Killer Moves", "Permanence", "Lead Change",
			"Completion", "Duration", "Final Score"]

def get_fen(position_num):
	with open("fen_positions.txt", "r") as file:
		line = next(islice(file, position_num, None), None)
		return line


def get_mirror_image(variant_num):
	fen = get_fen(variant_num)
	string = fen[0:8][::-1]
	i = 0
	with open("fen_positions.txt", "r") as file:
		for line in file:
			if string in line:
				return i
			i += 1



def uncertainty(eval_co):
	S = 100
	uncertainty = 0
	for s in range(S):
		t = s/S #***** not s/(S-1) because of indexing
		
		ind = int(t*len(eval_co))
		sm = eval_co[ind]
			
		uncertainty += min(1,t-sm)
	return uncertainty / S
			
		
def killer_moves(eval_co):
	move_diff = [eval_co[i+1] + eval_co[i] for i in range(len(eval_co)-2)]
	return max(move_diff)

def permanence(eval_co):

	inner = 0
	num_triples = len(eval_co)-2
	for i in range(num_triples):
		inner += eval_co[i+2]+eval_co[i+1]+eval_co[i+1]+eval_co[i]
	return 1- inner/num_triples

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

	

def lead_change(eval_co):
	leader = get_leads(eval_co)
	return np.mean([leader[i] != leader[i+1] for i in range(len(leader)-1)])


def completion(outcome):
	return outcome != 0

def duration(eval_co,m_pref):

	return abs(m_pref-len(eval_co))/m_pref


def get_single_statistics():
	filename = "filtered_data.txt"
	# print(read_last_line(filename))
	output_name = "single_data.txt"
	counter = 0
	ordered_weights = np.array([0.2023,0.3585,0.1027,-0.2769,0.0788,-0.0907])
	ordered_weights = ordered_weights/np.sum(ordered_weights)
	with open(filename, 'r') as file:
		for fen_number in range(960): # should be NUM_POSITIONS
			evaluations,outcomes = get_data(list(islice(file, NUM_LINES)),counter)
			counter += NUM_LINES
			stats = [[] for _ in range(6)]
			for i in range(len(evaluations)):
				
				eval_co = evaluations[i]
				m_pref = np.mean([len(evals) for evals in evaluations])
				stats[0].append(uncertainty(eval_co))
				stats[1].append(killer_moves(eval_co))
				stats[2].append(permanence(eval_co))
				stats[3].append(lead_change(eval_co))
				stats[4].append(completion(outcomes[i]))
				stats[5].append(duration(eval_co,m_pref))
	
			stats = np.array(stats)
			stats = stats.T
			final_scores = np.matmul(stats,ordered_weights)
			data = np.c_[stats, final_scores]
			print(data.shape)
			lines = []
			for row in data:
				lines.append(f"{fen_number}_{'_'.join([str(r) for r in row])}\n")
			with open(output_name,'a') as output:
				output.writelines(lines)


def load_single_data():
	filename = "single_data.txt"
	data = [[] for _ in range(960)]
	with open(filename,'r') as file:
		for line in file:
			l = line[:-1].split("_")
			ind = int(l[0])
			rest = [float(i) for i in l[1:]]
			data[ind].append(rest)
	return data

def generate_all_mirrored_pairs():
	pairs = []
	for i in range(960):
		j = get_mirror_image(i)
		if i < j:
			pairs.append((i,j))
	return pairs


def get_p_vals():
	data = load_single_data()
	pairs = generate_all_mirrored_pairs()
	all_p_vals = [None for i in range(960)]
	for tup in pairs:
		i,j = tup
		p_vals = []
		for k in range(7):
			t_stat, p_value = ttest_ind(data[i][:][k], data[j][:][k])
			p_vals.append(p_value)
		all_p_vals[i] = p_vals
		all_p_vals[j] = p_vals
	all_p_vals = np.array(all_p_vals)
	print(all_p_vals.shape)

	np.savez("matthew_p_vals.npz",arr1 = all_p_vals)
	

def analyze_p_vals():
	
	data = np.load("matthew_p_vals.npz",allow_pickle=True)
	data = data['arr1']
	maxes = np.max(data, axis=0)
	means = np.mean(data, axis=0)
	mins = np.min(data, axis=0)

	# print(maxes)
	# print(means)
	# print(mins)
	for i in range(7):
		print(f'\\item \\textbf{{{names[i]}}}: Max p-val: {round(maxes[i],4)}, Mean p-val: {round(means[i],4)}, Min p-val: {round(mins[i],4)}')
		# print(f'{names[i]}: Max p-val: {round(maxes[i],4)}, Mean p-val: {round(means[i],4)}, Min p-val: {round(mins[i],4)}')


def get_individual_stds():
	data = load_single_data()
	stds = [None for i in range(960)]

	pairs = generate_all_mirrored_pairs()
	for tup in pairs:
		i,j = tup
		trials = data[i]+data[j]
		std_dev = np.std(trials,axis =0)
		stds[i] = std_dev
		stds[j] = std_dev
	stds = np.array(stds)
	# print(stds.shape)
	stds = stds / np.sqrt(750)
	return stds

def get_abs_diff_arr():
	raw = np.load("matthew_final.npz",allow_pickle=True)
	raw = raw['arr1']
	pairs = generate_all_mirrored_pairs()
	diffs = [None for i in range(960)]
	for tup in pairs:
		i,j = tup
		new = np.abs(raw[i]-raw[j])
		diffs[i] = new
		diffs[j] = new
	diffs = np.array(diffs)
	return diffs


def compare_stds_to_diff_means():
	stds = get_individual_stds()
	diffs = get_abs_diff_arr()
	data = diffs/stds
	maxes = np.max(data, axis=0)
	means = np.mean(data, axis=0)
	mins = np.min(data, axis=0)

	# print(maxes)
	# print(means)
	# print(mins)
	for i in range(7):
		print(f'\\item \\textbf{{{names[i]}}}: Max: {round(maxes[i],4)}, Mean: {round(means[i],4)}, Min: {round(mins[i],4)}')
		# print(f'{names[i]}: Max: {round(maxes[i],4)}, Mean: {round(means[i],4)}, Min: {round(mins[i],4)}')

	
	



	

	

compare_stds_to_diff_means()
# analyze_p_vals()

	

	
	
	
