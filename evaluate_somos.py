import pandas as pd
import os
from tqdm import tqdm
from gpt4o_audio_api import ab_testing, gpt4o_mos

def process_audio_scores(input_csv, output_csv, audio_path, model_function):
    """
    Processes audio files listed in the input CSV, evaluates them using a provided model function, and writes results to an output CSV.

    Parameters:
        input_csv (str): Path to the input CSV file containing audio file names.
        output_csv (str): Path to the output CSV file to save results.
        audio_path (str): Directory containing the audio files.
        model_function (callable): Function to evaluate an audio file and return a score.
    """
    # Load the input CSV
    df = pd.read_csv(input_csv)
    file_list = df['utteranceId'].to_numpy().tolist()
    i=0
    # Open the output CSV for writing
    with open(output_csv, 'w', encoding='utf8') as csv:
        csv.write('filename,gpt4o-score\n')
        for file in tqdm(file_list):
            file_path = os.path.join(audio_path, file)

            if os.path.exists(file_path):
                print("File path exists.")
            else:
                print(f"File path does not exist: {file_path}")
                continue

            if i==0:
                output = model_function(file_path)
                csv.write(f'{file_path},{output}\n')
            i+=1

def ab_testing_and_write_results(audio_path, input_csv, output_file, reference_audio_file, ab_testing_fn):
    """
    Generalized function for AB testing and writing results to a file.

    Parameters:
    - audio_path (str): Path to the directory containing audio files.
    - input_csv (str): Path to the input CSV file containing filenames to compare with the reference file.
    - output_file (str): Path to the output file where results will be written.
    - reference_audio_file (str): The reference audio file for AB testing.
    - ab_testing_fn (function): Function to perform AB testing between two audio files.

    Returns:
    None
    """
    # Construct the path to the reference audio file
    reference_audio_path = os.path.join(audio_path, reference_audio_file)
    
    # Load the input CSV and extract the list of audio files to compare
    df = pd.read_csv(input_csv)
    file_list = df['utteranceId'].tolist()

    # Check if the reference audio file exists
    if not os.path.exists(reference_audio_path):
        print(f"Reference file '{reference_audio_file}' path does not exist.")
        return

    print(f"Reference file '{reference_audio_file}' path exists.")
    
    # Open the output file for writing the results
    with open(output_file, 'w', encoding='utf-8') as tsv:
        tsv.write('Reference_Audio\tComparison_Audio\tAssessment\n')
        i=0
        # Iterate through the list of comparison files and perform AB testing
        for file in tqdm(file_list, desc="Processing files"):
            comparison_audio_path = os.path.join(audio_path, file)

            # Check if the comparison audio file exists
            if not os.path.exists(comparison_audio_path):
                print(f"Comparison file '{file}' path does not exist.")
                pdb.set_trace()  # Debugging stop point
                continue  # Skip the rest of the iteration for this file

            print(f"Comparison file '{file}' path exists.")
            if i==0:
                # Perform AB testing and get the assessment result
                assessment = ab_testing_fn(reference_audio_path, comparison_audio_path)

                # Write the results to the output file
                tsv.write(f'{reference_audio_path}\t{comparison_audio_path}\t{assessment}\n')
            i+=1

test_mos_list = '/data/share/data/Speech/somos/training_files/split1/clean/test_mos_list.csv'

process_audio_scores(test_mos_list, 'MOS-somos_gpt4o-audio.csv', '/data/share/data/Speech/somos/audios/', gpt4o_mos)

ab_testing_and_write_results(
     audio_path='/data/share/data/Speech/somos/audios',
     input_csv=test_mos_list,
     output_file='AB-Testing_somos.tsv',
     reference_audio_file='LJ017-0230_009.wav',
     ab_testing_fn=ab_testing
)