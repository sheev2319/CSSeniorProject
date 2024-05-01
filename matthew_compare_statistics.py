import numpy as np
from itertools import islice

UNCERTAINTY = 0
KILLER_MOVES = 1
PERMANENCE = 2
LEAD_CHANGE = 3
COMPLETION = 4
DURATION = 5
FINAL_SCORE = 6

NORMAL_CHESS = 518


data = np.load("matthew_statistics.npz",allow_pickle=True)
data = data['arr1']

names=  ["Uncertainty", "Killer Moves", "Permenance", "Lead Change",
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

# print("bqnnrbkr/pppppppp/8/8/8/8/PPPPPPPP/BQNNRBKR w KQkq - 0 1")
# print(get_fen(2))
# mirror = get_mirror_image(2)
# print(mirror)
# print(get_fen(mirror))
# for i in range(960):
# 	if get_mirror_image(get_mirror_image(i))!= i:
# 		print(i)
			
for i in range(7):
	mirror_image_diffs = [abs(data[variant][i] - data[get_mirror_image(variant)][i]) for variant in range(960)]
	# print(names[i])
	# print(np.mean(mirror_image_diffs))
	# print(np.std(mirror_image_diffs))
	# print(np.mean(mirror_image_diffs)/np.mean(data[:,i]))
	# print(np.std(mirror_image_diffs)/np.mean(data[:,i]),end = "\n\n")
	print(f'\\textbf{{{names[i]}}}',end = ": ")
	print(f'Diff Mean: {round(np.mean(mirror_image_diffs),4)}',end = ", ")
	print(f'Diff Std Dev: {round(np.std(mirror_image_diffs),4)}',end = ", ")
	print(f'Sample Mean: {round(np.mean(data[:,i]),4)}')


'''
arr = [[(data[variant][i] + data[get_mirror_image(variant)][i])/2 for i in range(7)] for variant in range(960)]

averaged_data = np.array(arr)
print(averaged_data.shape)
np.savez("matthew_statistics_averaged.npz",arr1 = averaged_data)
	
'''