import numpy as np
import matplotlib.pyplot as plt
import leo_read_games
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

def get_mirror_image(variant_num):
    fen = leo_read_games.get_fen(variant_num)
    string = fen[0:8][::-1]
    i = 0
    with open("fen_positions.txt", "r") as file:
        for line in file:
            if string in line:
                return i
            i += 1

STANDARD_CHESS = 518 # variant number of normal chess

leo_data = np.load('entropies_and_outcomes.npz', allow_pickle=True)
matthew_data = np.load('matthew_final_averaged.npz', allow_pickle=True)
matthew_results = matthew_data['arr1']
ply_entropies = leo_data['arr1']
outcomes = leo_data['arr2']
white_win_rates = np.array([variant[0]/(variant[0]+variant[1]+variant[2]) for variant in outcomes])
draw_rates = np.array([variant[1]/(variant[0]+variant[1]+variant[2]) for variant in outcomes])
black_win_rates = np.array([variant[2]/(variant[0]+variant[1]+variant[2]) for variant in outcomes])

matthew_final_score = [matthew_results[variant][5] for variant in range(960)]

ply_20_entropies = np.array([float(variant[19]) for variant in ply_entropies])
ply_40_entropies = np.array([float(variant[39]) for variant in ply_entropies])

ply_40_entropies = [(ply_40_entropies[variant] + ply_40_entropies[get_mirror_image(variant)])/2 for variant in range(960)]
ply_20_entropies = [(ply_20_entropies[variant] + ply_20_entropies[get_mirror_image(variant)])/2 for variant in range(960)]
white_win_rates = np.array([(white_win_rates[variant] + white_win_rates[get_mirror_image(variant)])/2 for variant in range(960)])
draw_rates = np.array([(draw_rates[variant] + draw_rates[get_mirror_image(variant)])/2 for variant in range(960)])
black_win_rates = np.array([(black_win_rates[variant] + black_win_rates[get_mirror_image(variant)])/2 for variant in range(960)])

def perform_ttest():
    full_array = []
    with open('ply40_negative_log_prob.txt', "r") as file:
        # Read each line from the file
        for line in file:
            # Strip newline characters and split the line by commas
            subarray = line.strip().split(',')
            # Convert each string in the subarray back into a float
            subarray = [float(num) for num in subarray]
            # Append the subarray of floats to the full array
            full_array.append(subarray)
    p_vals = [0 for i in range(960)]
    for variant in range(960):
        t_stat, p_val = ttest_ind(full_array[variant], full_array[get_mirror_image(variant)])
        p_vals[variant] = p_val
    p_vals = np.array(p_vals)
    mean_pval = np.mean(p_vals)
    max_pval = max(p_vals)
    min_pval = min(p_vals)
    median_pval = np.median(p_vals)
    print(f"Mean p value is {mean_pval}, max p value is {max_pval}, min p value is {min_pval}, median p value is {median_pval}")

def highest_lowest(arr, n):
    """Returns the indices for the n highest and n lowest values"""
    indices_highest = np.argsort(arr)[-n:][::-1] # sorted from highest to lowest
    indices_lowest = np.argsort(arr)[:n] # sorted from lowest to highest
    return indices_highest, indices_lowest

def plot_entropy_histograms():
    # plot ply20 and ply40 entropy histograms
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plotting histograms
    axs[0].hist(ply_20_entropies, bins=30, color='#aec6cf')
    axs[1].hist(ply_40_entropies, bins=30, color='#77dd77')

    # Setting titles and labels
    axs[0].set_title('Histogram of ply 20 entropies')
    axs[1].set_title('Histogram of ply 40 entropies')
    axs[0].set_xlabel('Entropy')
    axs[1].set_xlabel('Entropy')
    axs[0].set_ylabel('Frequency')
    axs[1].set_ylabel('Frequency')
    
    axs[0].axvline(x=ply_20_entropies[STANDARD_CHESS], color='r', linestyle='--', linewidth=2, label='Standard chess')
    axs[1].axvline(x=ply_40_entropies[STANDARD_CHESS], color='r', linestyle='--', linewidth=2, label='Standard chess')
    
    axs[0].legend()
    axs[1].legend()
    plt.show()
    return

def plot_outcome_histograms():
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plotting histograms
    axs[0].hist(white_win_rates, bins=20, color='#5fa777')
    axs[1].hist(draw_rates, bins=20, color='#ff6f61')
    axs[2].hist(black_win_rates, bins=20, color='#4a536b')

    # Setting titles and labels
    axs[0].set_title('White win rates')
    axs[1].set_title('Draw rates')
    axs[2].set_title('Black win rates')
    axs[0].set_xlabel('Rate')
    axs[1].set_xlabel('Rate')
    axs[2].set_xlabel('Rate')
    axs[0].set_ylabel('Frequency')
    axs[1].set_ylabel('Frequency')
    axs[2].set_ylabel('Frequency')
    
    axs[0].axvline(x=white_win_rates[STANDARD_CHESS], color='r', linestyle='--', linewidth=2, label='Standard chess')
    axs[1].axvline(x=draw_rates[STANDARD_CHESS], color='r', linestyle='--', linewidth=2, label='Standard chess')
    axs[2].axvline(x=black_win_rates[STANDARD_CHESS], color='r', linestyle='--', linewidth=2, label='Standard chess')    
    
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.show()
    return

def find_correlations():
    correlation_matrix_draw_ply40entropy = np.corrcoef(draw_rates, ply_40_entropies)
    correlation_matrix_whitewinminusblackin_ply40entropy = np.corrcoef(black_win_rates-white_win_rates, ply_40_entropies)
    correlation_matrix_threefoldrep_ply40entropy = np.corrcoef(leo_read_games.num_threefold_repetitions_per_variant, ply_40_entropies)
    correlation_matrix_rookspacing_ply40entropy = np.corrcoef(np.array([count_spaces_between_pieces(variant, 'r') for variant in range(960)]), ply_40_entropies)
    correlation_matrix_kingposition_ply40entropy = np.corrcoef(np.abs(np.array([piece_position(variant, 'k') for variant in range(960)])-3.5), ply_40_entropies)
    correlation_matrix_bishopspacing_ply40entropy = np.corrcoef(np.array([count_spaces_between_pieces(variant, 'b') for variant in range(960)]), ply_40_entropies)
    correlation_matrix_matthewfinalscore_ply40entropy = np.corrcoef(matthew_final_score, ply_40_entropies)
    correlation_matrix_whitewinrate_ply40entropy = np.corrcoef(white_win_rates, ply_40_entropies)
    correlation_matrix_queenposition_ply40entropy = np.corrcoef(np.abs(np.array([piece_position(variant, 'k') for variant in range(960)])-3.5), ply_40_entropies)
    correlation_matrix_knight_ply40entropy = np.corrcoef(np.array([count_spaces_between_pieces(variant, 'n') for variant in range(960)]), ply_40_entropies)
    correlation_matrix_bishopscore_ply40entropy = np.corrcoef(np.array([bishop_placing_score(variant) for variant in range(960)]), ply_40_entropies)
    
    correlation_draw_ply40entropy = correlation_matrix_draw_ply40entropy[0,1]
    correlation_whitewinminusblackwin_ply40entropy = correlation_matrix_whitewinminusblackin_ply40entropy[0,1]
    correlation_threefoldrep_ply40entropy = correlation_matrix_threefoldrep_ply40entropy[0,1]
    correlation_rookspacing_ply40entropy = correlation_matrix_rookspacing_ply40entropy[0,1]
    correlation_kingposition_ply40entropy = correlation_matrix_kingposition_ply40entropy[0,1]
    correlation_bishopspacing_ply40entropy = correlation_matrix_bishopspacing_ply40entropy[0,1]
    correlation_matthewfinalscore_ply40entropy = correlation_matrix_matthewfinalscore_ply40entropy[0,1]
    correlation_whitewinrate_ply40entropy = correlation_matrix_whitewinrate_ply40entropy[0,1]
    correlation_queenposition_ply40entropy = correlation_matrix_queenposition_ply40entropy[0,1]
    correlation_knight_ply40entropy = correlation_matrix_knight_ply40entropy[0,1]
    correlation_bishopscore_ply40entropy = correlation_matrix_bishopscore_ply40entropy[0,1]
    
    print(f"Correlation between draw rate and ply 40 entropy: {correlation_draw_ply40entropy}")
    print(f"Correlation between white win rate minus black win rate and ply 40 entropy: {correlation_whitewinminusblackwin_ply40entropy}")
    print(f"Correlation between number of threefold repetitions observed and ply 40 entropy: {correlation_threefoldrep_ply40entropy}")
    print(f"Correlation between spacing between rooks and ply 40 entropy: {correlation_rookspacing_ply40entropy}")
    print(f"Correlation between king position and ply 40 entropy: {correlation_kingposition_ply40entropy}")
    print(f"Correlation between spacing between bishops and ply 40 entropy: {correlation_bishopspacing_ply40entropy}")
    print(f"Correlation between Matthew final score and ply 40 entropy: {correlation_matthewfinalscore_ply40entropy}")
    print(f"Correlation between white win rate and ply 40 entropy is: {correlation_whitewinrate_ply40entropy}")
    print(f"Correlation between queen position and ply 40 entropy is: {correlation_queenposition_ply40entropy}")
    print(f"Correlation between knight spacing and ply 40 entropy is: {correlation_knight_ply40entropy}")
    print(f"Correlation between bishop placing score and ply 40 entropy is: {correlation_bishopscore_ply40entropy}")
    
def count_spaces_between_pieces(variant_num, piece):
    fen_string = leo_read_games.get_fen(variant_num)
    first_index = fen_string.find(piece)
    second_index = fen_string.find(piece, first_index + 1)
    num_chars_between = second_index - first_index - 1
    return num_chars_between

def piece_position(variant_num, piece):
    fen_string = leo_read_games.get_fen(variant_num)
    first_k_index = fen_string.find(piece)
    return first_k_index

def bishop_positions(variant_num):
    fen_string = leo_read_games.get_fen(variant_num)
    first_index = fen_string.find('b')
    second_index = fen_string.find('b', first_index + 1)
    return (first_index, second_index)

def bishop_placing_score(variant_num):
    """Bishops closer to the corners have a higher score"""
    first_index, second_index = bishop_positions(variant_num)
    scores = [4,3,2,1,1,2,3,4]
    return scores[first_index] + scores[second_index]

def get_unique_representation(variant_num):
    fen = leo_read_games.get_fen(variant_num)
    string = fen[0:8][::-1]
    if string == fen:
        return variant_num
    i = 0
    with open("fen_positions.txt", "r") as file:
        for line in file:
            if string in line:
                return min(variant_num, i)
            i += 1

def scatter_plots():
    plt.scatter(ply_40_entropies, white_win_rates, c='#4a69bd', marker='o')  # 'c' is the color, 'marker' specifies the shape of the dots
    plt.title('Scatter plot of ply 40 entropy vs white win rate')
    plt.xlabel('Ply 40 entropy')
    plt.ylabel('White win rate')
    plt.grid(True)  # Optional, adds a grid to the plot
    plt.show()

def decimals_to_floats(array):
    return [float(elt) for elt in array]
    
def plot_entropy_evaluation():
    highest_ply40entropies, lowest_ply40entropies = highest_lowest(ply_40_entropies, 6)
    fig, axs = plt.subplots(2, 3, figsize=(17, 10), sharey=True)
    axs[0][0].plot(decimals_to_floats(ply_entropies[highest_ply40entropies[0]]), color = '#00BFFF')
    axs[0][1].plot(decimals_to_floats(ply_entropies[highest_ply40entropies[2]]), color = '#9966CC')
    axs[0][2].plot(decimals_to_floats(ply_entropies[highest_ply40entropies[4]]), color = '#FF7F50')
    axs[1][0].plot(decimals_to_floats(ply_entropies[lowest_ply40entropies[0]]), color = '#800000')
    axs[1][1].plot(decimals_to_floats(ply_entropies[lowest_ply40entropies[2]]), color = '#91C086')
    axs[1][2].plot(decimals_to_floats(ply_entropies[lowest_ply40entropies[4]]), color = '#FDBCB4')
    
    axs[0][0].set_title('Entropy plot of highest entropy variant (741)')
    axs[0][1].set_title('Entropy plot of second-highest entropy variant (633)')
    axs[0][2].set_title('Entropy plot of third-highest entropy variant (153)')
    axs[1][0].set_title('Entropy plot of lowest entropy variant (176)')
    axs[1][1].set_title('Entropy plot of second-lowest entropy variant (376)')
    axs[1][2].set_title('Entropy plot of third-lowest entropy variant (479)')
    axs[0][0].set_xlabel('Ply')
    axs[0][1].set_xlabel('Ply')
    axs[0][2].set_xlabel('Ply')
    axs[1][0].set_xlabel('Ply')
    axs[1][1].set_xlabel('Ply')
    axs[1][0].set_xlabel('Ply')
    axs[0][0].set_ylabel('Entropy')
    axs[0][1].set_ylabel('Entropy')
    axs[0][2].set_ylabel('Entropy')
    axs[1][0].set_ylabel('Entropy')
    axs[1][1].set_ylabel('Entropy')
    axs[1][2].set_ylabel('Entropy')
    
    plt.show()
    
def get_distinct_variants():
    distinct_variants = []
    for variant_num in range(960):
        distinct_variants.append(get_unique_representation(variant_num))
    return set(distinct_variants)

def king_position_entropy_box_plot():
    king_positions = np.floor(np.abs(np.array([piece_position(variant, 'k') for variant in range(960)]) - 3.5))
    kingpos_entropy = pd.DataFrame({
        'King position': king_positions,
        'Ply 40 entropy': ply_40_entropies
    })
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='King position', y='Ply 40 entropy', data=kingpos_entropy, palette='deep')
    plt.title('Entropy Distribution by King Position')
    plt.xlabel('King distance from center')
    plt.ylabel('Ply 40 Entropy')
    plt.show()
  
def rook_position_entropy_box_plot():
    rook_positions = np.array([count_spaces_between_pieces(variant, 'r') for variant in range(960)])
    rookpos_entropy = pd.DataFrame({
        'Rook position': rook_positions,
        'Ply 40 entropy': ply_40_entropies
    })
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Rook position', y='Ply 40 entropy', data=rookpos_entropy, palette='muted')
    plt.title('Entropy Distribution by Distance Between Rooks')
    plt.xlabel('# squares between rooks')
    plt.ylabel('Ply 40 Entropy')
    plt.show() 

def queen_position_entropy_box_plot():
    queen_positions = np.floor(np.abs(np.array([piece_position(variant, 'q') for variant in range(960)]) - 3.5))
    queenpos_entropy = pd.DataFrame({
        'Queen position': queen_positions,
        'Ply 40 entropy': ply_40_entropies
    })
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Queen position', y='Ply 40 entropy', data=queenpos_entropy, palette='Blues')
    plt.title('Entropy Distribution by Queen Position')
    plt.xlabel('Queen distance from center')
    plt.ylabel('Ply 40 Entropy')
    plt.show()

def matthew_stats_entropy_scatter():
    fig, axs = plt.subplots(2, 3, figsize=(17, 10), sharey=True)
    axs[0][0].scatter([matthew_results[variant][0] for variant in range(960)], ply_40_entropies, color = '#00BFFF')
    axs[0][1].scatter([matthew_results[variant][1] for variant in range(960)], ply_40_entropies, color = '#9966CC')
    axs[0][2].scatter([matthew_results[variant][2] for variant in range(960)], ply_40_entropies, color = '#FF7F50')
    axs[1][0].scatter([matthew_results[variant][3] for variant in range(960)], ply_40_entropies, color = '#800000')
    axs[1][1].scatter([matthew_results[variant][4] for variant in range(960)], ply_40_entropies, color = '#91C086')
    axs[1][2].scatter([matthew_results[variant][5] for variant in range(960)], ply_40_entropies, color = '#FDBCB4')
    
    axs[0][0].set_title('Uncertainty vs ply 40 entropy')
    axs[0][1].set_title('Killer moves vs ply 40 entropy')
    axs[0][2].set_title('Permanence vs ply 40 entropy')
    axs[1][0].set_title('Lead change vs ply 40 entropy')
    axs[1][1].set_title('Completion vs ply 40 entropy')
    axs[1][2].set_title('Duration vs ply 40 entropy')
    
    axs[0][0].set_xlabel('Uncertainty')
    axs[0][1].set_xlabel('Killer moves')
    axs[0][2].set_xlabel('Permanence')
    axs[1][0].set_xlabel('Lead change')
    axs[1][1].set_xlabel('Completion')
    axs[1][2].set_xlabel('Duration')
    
    axs[0][0].set_ylabel('Entropy')
    axs[0][1].set_ylabel('Entropy')
    axs[0][2].set_ylabel('Entropy')
    axs[1][0].set_ylabel('Entropy')
    axs[1][1].set_ylabel('Entropy')
    axs[1][2].set_ylabel('Entropy')
    
    plt.show() 
    
#rook_position_entropy_box_plot()
#plot_entropy_histograms()
#plot_outcome_histograms()
#find_correlations()
#highest_ply40entropies, lowest_ply40entropies = highest_lowest(ply_40_entropies, 6)
#print(f"The variants with highest ply 40 entropies were (highest to lowest): {highest_ply40entropies}")
#print(f"The variants with lowest ply 40 entropies were (lowest to highest): {lowest_ply40entropies}")

#print(ply_40_entropies[highest_ply40entropies[0]])
#print(ply_40_entropies[highest_ply40entropies[1]])
#print(ply_40_entropies[highest_ply40entropies[2]])
#print(ply_40_entropies[highest_ply40entropies[3]])
#print(ply_40_entropies[highest_ply40entropies[4]])
#print(ply_40_entropies[highest_ply40entropies[5]])

#mirror_image_entropy_diffs = [abs(ply_40_entropies[variant] - ply_40_entropies[get_mirror_image(variant)]) for variant in range(960)]
#print(np.mean(mirror_image_entropy_diffs))
#print(np.std(mirror_image_entropy_diffs))
#print(min(ply_40_entropies))
#print(max(ply_40_entropies))


#print(f"mean/std: {np.std(mirror_image_entropy_diffs)/np.mean(ply_40_entropies)}")

#scatter_plots()
#matthew_stats_entropy_scatter()
#plot_entropy_evaluation()
#print(matthew_final_score)
#queen_position_entropy_box_plot()
#print(ply_40_entropies[88])
#rook_position_entropy_box_plot()
#perform_ttest()