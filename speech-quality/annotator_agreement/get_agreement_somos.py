import pandas as pd
import json
import numpy as np
from tqdm import tqdm  # For progress reporting

def process_agreement(json_file, somos_root_path, data_path, output_dir="./"):
    """
    Process pairwise agreement from a test dataset and a JSON file of utterance pairs.

    Parameters:
        json_file (str): Name of the JSON file (without extension).
        somos_root_path (str): Path to the folder containing the test dataset.
        data_path (str): Path to the JSON file containing utterance pairs.
        output_dir (str): Directory to save the output CSV file. Default is the current directory.

    Output:
        A CSV file with the agreement results.
    """
    # Dataset files
    dataset = [
        'raw_scores_removed_excess_gt_testset.tsv',
        'raw_scores_removed_excess_gt_trainset.tsv',
        'raw_scores_removed_excess_gt_validset.tsv'
    ]
    testset = somos_root_path + dataset[0]

    # Load the test dataset
    try:
        df = pd.read_csv(testset, sep='\t', encoding='utf8')
    except FileNotFoundError:
        print(f"Error: Test dataset file {testset} not found.")
        return

    # Load the JSON data
    try:
        df_json = pd.read_json(data_path, encoding='utf8')
        list_json = df_json.to_numpy().tolist()
    except FileNotFoundError:
        print(f"Error: JSON file {data_path} not found.")
        return

    # Prepare output file
    output_file = f"{output_dir}/agreement_{json_file}.csv"
    with open(output_file, 'w', encoding='utf8') as agm:
        agm.write('A_utterance,B_utterance,MaxAgree,TotalNonEqual,Acc\n')

        overall_max_agree = 0
        overall_total_non_equal = 0

        # Process each pair of utterances
        for a, b in tqdm(list_json, desc="Processing utterance pairs"):
            uttId_a = a.get('uttId')
            uttId_b = b.get('uttId')

            if not uttId_a or not uttId_b:
                print(f"Warning: Missing uttId for pair {a}, {b}. Skipping.")
                continue

            # Retrieve 'choice' arrays for both utterances
            np_a_choice = df[df['utteranceId'] == uttId_a]['choice'].to_numpy()
            np_b_choice = df[df['utteranceId'] == uttId_b]['choice'].to_numpy()

            if len(np_a_choice) == 0 or len(np_b_choice) == 0:
                print(f"Warning: No data for {uttId_a} or {uttId_b}. Skipping.")
                continue

            # Calculate counts for agreement
            count_a_greater_b = np.sum(np_a_choice[:, None] > np_b_choice)
            count_b_greater_a = np.sum(np_b_choice[:, None] > np_a_choice)
            count_non_equal = np.sum(np_a_choice[:, None] != np_b_choice)

            if count_non_equal > 0:
                max_agree = max(count_a_greater_b, count_b_greater_a)
                acc = max_agree * 100 / count_non_equal
                overall_max_agree += max_agree
                overall_total_non_equal += count_non_equal

                # Write results for this pair
                agm.write(f'{uttId_a},{uttId_b},{max_agree},{count_non_equal},{acc:.2f}\n')
            else:
                print(f"Warning: No non-equal comparisons for {uttId_a}, {uttId_b}. Skipping.")

    # Output overall agreement
    if overall_total_non_equal > 0:
        overall_acc = overall_max_agree * 100 / overall_total_non_equal
        print(json_file)
        print(f'Overall Agreement (%) = {overall_acc:.2f}')
    else:
        print("No utterance pairs were processed.")

process_agreement(
    json_file="data_somos_pairwise_diff15",
    somos_root_path="/data/share/data/Speech/somos/raw_scores_with_metadata/split1/",
    data_path="/data/workspace/ppotsawee/audioLM-as-judge/data/data_somos_pairwise_diff15.json",
    output_dir="./"
)

process_agreement(
    json_file="data_somos_pairwise_diff1",
    somos_root_path="/data/share/data/Speech/somos/raw_scores_with_metadata/split1/",
    data_path="/data/workspace/ppotsawee/audioLM-as-judge/data/data_somos_pairwise_diff1.json",
    output_dir="./"
)
process_agreement(
    json_file="data_somos_pairwise_diffall",
    somos_root_path="/data/share/data/Speech/somos/raw_scores_with_metadata/split1/",
    data_path="/data/workspace/ppotsawee/audioLM-as-judge/data/data_somos_pairwise_diffall.json",
    output_dir="./"
)