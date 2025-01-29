import pandas as pd
import numpy as np
from tqdm import tqdm  

def extract_scores(df, utt_id):
    """Extract quality and intelligibility scores for a given utterance ID."""
    row = df[df['filename'] == utt_id]
    if row.empty:
        return np.array([]), np.array([])  # Return empty arrays to handle missing data

    quality = row['Q'].values
    intell = row['I'].values
    return quality, intell

def calculate_agreement(np_a, np_b):
    """Compute agreement metrics between two numpy arrays."""
    if np_a.size == 0 or np_b.size == 0:
        return 0, 0  # No valid data to compare

    count_a_greater_b = np.sum(np_a[:, None] > np_b)
    count_b_greater_a = np.sum(np_b[:, None] > np_a)
    count_non_equal = np.sum(np_a[:, None] != np_b)

    if count_non_equal == 0:
        return 0, 0  # Avoid division by zero

    max_agree = max(count_a_greater_b, count_b_greater_a)
    return max_agree, count_non_equal

def process_agreement(json_file, csv_file, data_path, output_dir="./"):
    """Process pairwise agreement from test dataset and JSON file of utterance pairs."""
    
    try:
        df_json = pd.read_json(data_path, encoding='utf8')
        list_json = df_json.to_numpy().tolist()
    except FileNotFoundError:
        print(f"Error: JSON file {data_path} not found.")
        return

    try:
        df_sheets = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file {csv_file} not found.")
        return

    output_file = f"{output_dir}/agreement_{json_file}.csv"
    
    # Initialize overall counters
    overall_agree_quality, overall_non_equal_quality = 0, 0
    overall_agree_intell, overall_non_equal_intell = 0, 0

    with open(output_file, 'w', encoding='utf8') as agm:
        agm.write('A_utterance,B_utterance,MaxAgree_Quality,TotalNonEqual_Quality,Acc_Quality,'
                  'MaxAgree_Intell,TotalNonEqual_Intell,Acc_Intell\n')

        for a, b in tqdm(list_json, desc="Processing utterance pairs"):
            uttId_a = a.get('path', '').split('/')[-1].split('.')[0]
            uttId_b = b.get('path', '').split('/')[-1].split('.')[0]

            quality_a, intell_a = extract_scores(df_sheets, uttId_a)
            quality_b, intell_b = extract_scores(df_sheets, uttId_b)

            max_agree_quality, non_equal_quality = calculate_agreement(np.array(quality_a), np.array(quality_b))
            max_agree_intell, non_equal_intell = calculate_agreement(np.array(intell_a), np.array(intell_b))

            # Compute agreement percentages if applicable
            acc_quality = (max_agree_quality * 100 / non_equal_quality) if non_equal_quality > 0 else 0
            acc_intell = (max_agree_intell * 100 / non_equal_intell) if non_equal_intell > 0 else 0

            # Update overall counters
            overall_agree_quality += max_agree_quality
            overall_non_equal_quality += non_equal_quality
            overall_agree_intell += max_agree_intell
            overall_non_equal_intell += non_equal_intell

            # Write to file
            agm.write(f'{uttId_a},{uttId_b},{max_agree_quality},{non_equal_quality},{acc_quality:.2f},'
                      f'{max_agree_intell},{non_equal_intell},{acc_intell:.2f}\n')

    # Final output for overall agreement
    if overall_non_equal_quality > 0:
        print(f"Overall Quality Agreement: {overall_agree_quality * 100 / overall_non_equal_quality:.2f}%")
    else:
        print("No valid quality comparisons available.")

    if overall_non_equal_intell > 0:
        print(f"Overall Intelligibility Agreement: {overall_agree_intell * 100 / overall_non_equal_intell:.2f}%")
    else:
        print("No valid intelligibility comparisons available.")

# Run the process
data_path = './data_tmhintqi_pairwise_diffall.json'
process_agreement('thaimos.csv', 'TMHINTQI_cleaned_raw_test.csv', data_path)
