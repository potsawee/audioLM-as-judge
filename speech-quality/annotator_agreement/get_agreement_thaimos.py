import pandas as pd
import numpy as np
from tqdm import tqdm  

def read_xls(excel_file):
    """Read all sheets from the Excel file into a dictionary of DataFrames."""
    sheets = pd.read_excel(excel_file, sheet_name=None)
    return {str(sheet): df for sheet, df in sheets.items()}

def extract_scores(df_sheets, utt_id):
    """Extract scores for a given utterance ID from all sheets."""
    quality, silence, pronunciation, avg = [], [], [], []
    for i in range(1, 17):
        df = df_sheets.get(str(i))
        if df is not None:
            row = df[df['ID'] == utt_id]
            if not row.empty:
                quality.append(row['Sound Quality'].values[0])
                silence.append(row['Silence'].values[0])
                pronunciation.append(row['Pronunciation'].values[0])
                avg.append((quality[-1] + silence[-1] + pronunciation[-1]) / 3)
    return quality, silence, pronunciation, avg

def process_agreement(json_file, excel_file, data_path, output_dir="./"):
    """Process pairwise agreement from test dataset and JSON file of utterance pairs."""
    
    try:
        df_json = pd.read_json(data_path, encoding='utf8')
        list_json = df_json.to_numpy().tolist()
    except FileNotFoundError:
        print(f"Error: JSON file {data_path} not found.")
        return

    # Read Excel sheets once to improve efficiency
    df_sheets = read_xls(excel_file)

    output_file = f"{output_dir}/agreement_{json_file}_pronunciation.csv"
    overall_max_agree, overall_total_non_equal = 0, 0

    with open(output_file, 'w', encoding='utf8') as agm:
        agm.write('A_utterance,B_utterance,MaxAgree,TotalNonEqual,Acc\n')
        overall_max_agree_quality = 0
        overall_total_non_equal_quality = 0
        overall_max_agree_silence = 0
        overall_total_non_equal_silence = 0
        overall_max_agree_pronunciation = 0
        overall_total_non_equal_pronunciation = 0

        overall_max_agree_avg = 0
        overall_total_non_equal_avg = 0
        for a, b in tqdm(list_json, desc="Processing utterance pairs"):
            uttId_a, uttId_b = a.get('datawow_id'), b.get('datawow_id')

            quality_a, silence_a, pronunciation_a, avg_a = extract_scores(df_sheets, uttId_a)
            quality_b, silence_b, pronunciation_b, avg_b = extract_scores(df_sheets, uttId_b)
            
            if not quality_a or not quality_b:
                print(f"Skipping {uttId_a}, {uttId_b}: Data missing.")
                continue

            # Calculate agreement for quality
            np_a_avg, np_b_avg = np.array(avg_a), np.array(avg_b)
            count_a_greater_b_avg = np.sum(np_a_avg[:, None] > np_b_avg)
            count_b_greater_a_avg = np.sum(np_b_avg[:, None] > np_a_avg)
            count_non_equal_avg = np.sum(np_a_avg[:, None] != np_b_avg)
            if count_non_equal_avg > 0:
                max_agree_avg = max(count_a_greater_b_avg, count_b_greater_a_avg)
                acc_avg = max_agree_avg * 100 / count_non_equal_avg
                overall_max_agree_avg += max_agree_avg
                overall_total_non_equal_avg += count_non_equal_avg


            # Calculate agreement for quality
            np_a_quality, np_b_quality = np.array(quality_a), np.array(quality_b)
            count_a_greater_b_quality = np.sum(np_a_quality[:, None] > np_b_quality)
            count_b_greater_a_quality = np.sum(np_b_quality[:, None] > np_a_quality)
            count_non_equal_quality = np.sum(np_a_quality[:, None] != np_b_quality)
            if count_non_equal_quality > 0:
                max_agree_quality = max(count_a_greater_b_quality, count_b_greater_a_quality)
                acc_quality = max_agree_quality * 100 / count_non_equal_quality
                overall_max_agree_quality += max_agree_quality
                overall_total_non_equal_quality += count_non_equal_quality

            # Calculate agreement for silence
            np_a_silence, np_b_silence = np.array(silence_a), np.array(silence_b)
            count_a_greater_b_silence = np.sum(np_a_silence[:, None] > np_b_silence)
            count_b_greater_a_silence = np.sum(np_b_silence[:, None] > np_a_silence)
            count_non_equal_silence = np.sum(np_a_silence[:, None] != np_b_silence)
            if count_non_equal_silence > 0:
                max_agree_silence = max(count_a_greater_b_silence, count_b_greater_a_silence)
                acc_silence = max_agree_silence * 100 / count_non_equal_silence
                overall_max_agree_silence += max_agree_silence
                overall_total_non_equal_silence += count_non_equal_silence

            # Calculate agreement for pronunciation
            np_a_pronunciation, np_b_pronunciation = np.array(pronunciation_a), np.array(pronunciation_b)
            count_a_greater_b_pronunciation = np.sum(np_a_pronunciation[:, None] > np_b_pronunciation)
            count_b_greater_a_pronunciation = np.sum(np_b_pronunciation[:, None] > np_a_pronunciation)
            count_non_equal_pronunciation = np.sum(np_a_pronunciation[:, None] != np_b_pronunciation)
            if count_non_equal_pronunciation > 0:
                max_agree_pronunciation = max(count_a_greater_b_pronunciation, count_b_greater_a_pronunciation)
                acc_pronunciation = max_agree_pronunciation * 100 / count_non_equal_pronunciation
                overall_max_agree_pronunciation += max_agree_pronunciation
                overall_total_non_equal_pronunciation += count_non_equal_pronunciation

        # Print or store the results as needed
        print(f"Average Score agreement: {overall_max_agree_avg*100/overall_total_non_equal_avg:.2f}%")
        
        print(f"Quality agreement: {overall_max_agree_quality*100/overall_total_non_equal_quality:.2f}%")
        print(f"Silence agreement: {overall_max_agree_silence*100/overall_total_non_equal_silence:.2f}%")
        print(f"Pronunciation agreement: {overall_max_agree_pronunciation*100/overall_total_non_equal_pronunciation:.2f}%")
# Run the process
data_path = './data_thaimos_pairwise_diffall.json'
excel_file = 'DataSheets_revised.xlsx'
process_agreement('thaimos.csv', excel_file, data_path)
