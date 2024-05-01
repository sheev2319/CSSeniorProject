import numpy as np
# import math
import matplotlib.pyplot as plt
from itertools import islice



data = np.load("matthew_final.npz",allow_pickle=True)
data = data['arr1']



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



def get_singles():
	singles = []
	for i in range(960):
		j = get_mirror_image(i)
		if i<j:
			singles.append((i,j))
	return singles


def get_diffs(doubled):
	singles = get_singles()
	diffs = []
	for tup in singles:
		i,j = tup
		diffs.append(data[i]-data[j])
		if doubled:
			diffs.append(data[j]-data[i])
	diffs = np.array(diffs)
	if not doubled:
		diffs = np.abs(diffs)
	print(diffs.shape)
	return diffs


def get_double_sided_diffs():
	diffs = []
	for i in range(960):
		j = get_mirror_image(i)
		diffs.append(data[i]-data[j])
	diffs = np.array(diffs)
	print(diffs.shape)
	return diffs



def get_graphs(diffs):
	names=  ["Uncertainty", "Killer Moves", "Permenance", "Lead Change",
			"Completion", "Duration", "Final Score"]
	colors = ["blue","red","green","purple","orange","brown","grey"]
	
	fig, axs = plt.subplots(3, 3, figsize=(9, 6))
	indexing_conversion = [0,1,2,3,4,5,7]
	for i in range(7):
		std_dev = get_std(diffs,i)
		j = indexing_conversion[i]
		indexing = (j//3,j%3)
		xs = [10*i for i in range(1,200)]
		ys = [std_dev/np.sqrt(x) for x in xs]
		axs[indexing].plot(xs, ys, color=colors[i], linestyle='-')
		axs[indexing].set_xlabel('Number of Trials')  # Label for the x-axis
		axs[indexing].set_ylabel(f"STD of {names[i]}")
		axs[indexing].set_title(f"Histogram of {names[i]}")
		# axs[indexing].set_xlim([0,1])
		

	axs[2, 0].axis('off')
	axs[2, 2].axis('off')

	# Adjust layout
	plt.tight_layout()
	plt.show()



def graph_variance(std_dev):

	xs = [10*i for i in range(1,200)]
	ys = [std_dev/np.sqrt(x) for x in xs]
	plt.plot(xs, ys, color='b', linestyle='-')
	plt.title('Standard Deviation vs Number of Trials')  # Add a title
	plt.xlabel('Number of Trials')  # Label for the x-axis
	plt.ylabel('Standard Deviation')  # Label for the y-axis

	plt.show()


def get_std(diffs,col_choice):
	return np.sqrt(750)*np.std(diffs[:,col_choice]) 


def print_std(diffs):
	names=  ["Uncertainty", "Killer Moves", "Permenance", "Lead Change",
			"Completion", "Duration", "Final Score"]
	for i in range(7):
		print(f'{names}: {round(get_std(diffs,i),4)}')

# graph_variance()
single_sided_diffs = get_diffs(doubled=False)
double_sided_diffs = get_diffs(doubled=True)

for i in range(7):
	a = get_std(single_sided_diffs,i)
	b = get_std(double_sided_diffs,i)
	print(f'{round(a,4):<10}{round(b,4)}')





# graph_variance(get_std(double_sided_diffs,ind))
# get_graphs(double_sided_diffs)


