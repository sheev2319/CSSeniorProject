import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import pandas as pd
import seaborn as sns

UNCERTAINTY = 0
KILLER_MOVES = 1
PERMANENCE = 2
LEAD_CHANGE = 3
COMPLETION = 4
DURATION = 5
FINAL_SCORE = 6

NORMAL_CHESS = 518



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

# data = []
# filename = "final_results_filtered.txt"
# with open(filename,'r') as file:
# 	for line in file:
# 		data.append([float(i) for i in line.split(" ")])

# data = np.array(data)
# np.savez("matthew_statistics.npz",arr1 = data)

# data = np.load("matthew_statistics.npz",allow_pickle=True)
# data = np.load("matthew_statistics_averaged.npz",allow_pickle=True)
# data = np.load("matthew_final.npz",allow_pickle=True)
data = np.load("matthew_final_averaged.npz",allow_pickle=True)
data = data['arr1']
print(data.shape)


def average_statistics(filename):

	d = np.array([(data[i]+data[get_mirror_image(i)])/2.0 for i in range(len(data))])
	# print(d.shape)
	np.savez(filename,arr1 = d)





def highest_lowest(arr, n,averaged = False):
	"""Returns the indices for the n highest and n lowest values"""
	if averaged:
		n*=2
	indices_highest = np.argsort(arr)[-n:][::-1] # sorted from highest to lowest
	indices_lowest = np.argsort(arr)[:n] # sorted from lowest to highest
	if averaged:
		return indices_highest[::2], indices_lowest[::2]
	else:
		return indices_highest, indices_lowest

def get_graphs():
	names=  ["Uncertainty", "Killer Moves", "Permenance", "Lead Change",
			"Completion", "Duration", "Final Score"]
	colors = ["blue","red","green","purple","orange","brown","grey"]
	alpha = 0.7
	bins = 30
	fig, axs = plt.subplots(3, 3, figsize=(9, 6))
	indexing_conversion = [0,1,2,3,4,5,7]
	for i in range(7):
		j = indexing_conversion[i]
		indexing = (j//3,j%3)
		axs[indexing].hist(data[:,i],bins=bins,color=colors[i],alpha=alpha)
		axs[indexing].axvline(x=np.mean(data[:,i]), color='k', linestyle='dashed', linewidth=1)
		axs[indexing].axvline(x=data[NORMAL_CHESS,i], color='k', linestyle='solid', linewidth=1)
		axs[indexing].set_title(f"Histogram of {names[i]}")
		# axs[indexing].set_xlim([0,1])
		

	axs[2, 0].axis('off')
	axs[2, 2].axis('off')

	# Adjust layout
	plt.tight_layout()
	plt.show()

# get_graphs()


# print(highest_lowest(data[:,-1],3,averaged=True))

def piece_distance(fen_code):
	dists = []
	with open("fen_positions.txt", "r") as file:
		for line in file:
			back_row = line.split("/")[0]
			ind1 = -1
			ind2 = -1
			for i in range(8):
				if back_row[i] == fen_code and ind1 == -1:
					ind1 = i
				elif back_row[i] == fen_code:
					ind2 = i
					continue
			dists.append(ind2-ind1)
	return np.array(dists)

def king_queen_distances():
	dists = []
	with open("fen_positions.txt", "r") as file:
		for line in file:
			back_row = line.split("/")[0]
			k = -1
			q = -1
			for i in range(8):
				if back_row[i] == "k":
					k=i
				elif back_row[i] == "q":
					q=i
			dists.append(1-min(abs(k-q)-1,1))
			# dists.append(abs(k-q))
	return np.array(dists)


def make_box_plot(x,y,x_name,y_name,title):
	df = pd.DataFrame({
		"x": x,
		"y": y
	})
	plt.figure(figsize=(10, 6))
	sns.boxplot(x='x', y='y', data=df, palette='muted')
	plt.title(title)
	plt.xlabel(x_name)
	plt.ylabel(y_name)
	plt.show() 




print(round(np.corrcoef(data[:,-1],piece_distance("b"))[0,1],4))
# print(np.corrcoef(data[:,-1],piece_distance("n"))[0,1])
print(round(np.corrcoef(data[:,-1],piece_distance("r"))[0,1],4))
print(round(np.corrcoef(data[:,-1],king_queen_distances())[0,1],4))

# make_box_plot(piece_distance("b"),data[:,-1],"Bishop Distance","Game Quality", "Bishop Distance vs Game Quality")
make_box_plot(king_queen_distances(),data[:,-1],"King Next to Queen","Game Quality", "King Next to Queen vs Game Quality")

# get_graphs()


# average_statistics("matthew_final_averaged.npz")

