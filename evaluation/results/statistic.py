import json
from os import name
import os.path
import pandas as pd

# Function to calculate the average score from a JSON file
def calculate_average_score(file_path):
    try:
        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Initialize variables for total score and count
        total_score = 0
        count = 0
        score_dict = {}
        # Iterate through the JSON data to sum scores
        for key, value in data.items():
            # Handle both direct float values and dict with "string_match" key
            if isinstance(value, dict):
                score = value.get("string_match", 0)
            elif isinstance(value, (int, float)):
                score = value
            else:
                score = 0
            score_dict[key] = score
            total_score += score
            count += 1

        # Calculate the average score
        average_score = total_score / count if count > 0 else 0
        score_dict['average_score'] = average_score
        # Print the results
        print(f"JSON Name: {file_path}")
        print(f"Average Score: {average_score:.2f}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print("Invalid JSON file.")
    return file_path, score_dict


def process_dataset(json_files, dataset_name):
    """Process a specific dataset and return the DataFrame"""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()} dataset")
    print(f"{'='*60}")
    
    pd_list = []
    all_keys = set()
    file_score_dicts = []
    
    # First pass: collect all keys and scores
    for file in json_files:
        file_name, score_dict = calculate_average_score(file)
        file_score_dicts.append((file_name, score_dict))
        all_keys.update(score_dict.keys())
    
    if not all_keys:
        print(f"No data found for {dataset_name}")
        return None
    
    # Sort keys for consistent column ordering, with 'average_score' at the end
    sorted_keys = sorted([k for k in all_keys if k != 'average_score'])
    if 'average_score' in all_keys:
        sorted_keys.append('average_score')
    
    # Build data rows
    for file_name, score_dict in file_score_dicts:
        score_list = [score_dict.get(key, None) for key in sorted_keys]
        score_list.insert(0, file_name)
        pd_list.append(score_list)
    
    columns = ['File Name'] + sorted_keys
    df = pd.DataFrame(pd_list, columns=columns)
    return df


if __name__ == '__main__':
    files = os.listdir('.')
    print('Files in directory:')
    print(files)
    
    # Filter and categorize JSON files
    json_files = [file for file in files if file.endswith('.json')]
    
    longbench_files = [f for f in json_files if 'longbench' in f.lower()]
    ruler_files = [f for f in json_files if 'ruler' in f.lower()]
    other_files = [f for f in json_files if f not in longbench_files and f not in ruler_files]
    
    print(f"\nFound {len(longbench_files)} LongBench files, {len(ruler_files)} RULER files, {len(other_files)} other files")
    
    # Process LongBench results
    if longbench_files:
        df_longbench = process_dataset(longbench_files, 'longbench')
        if df_longbench is not None:
            output_file = 'longbench_scores.xlsx'
            df_longbench.to_excel(output_file, index=False)
            print(f"\nLongBench results saved to '{output_file}'")
    else:
        print("\nNo LongBench JSON files found")
    
    # Process RULER results
    if ruler_files:
        df_ruler = process_dataset(ruler_files, 'ruler')
        if df_ruler is not None:
            output_file = 'ruler_scores.xlsx'
            df_ruler.to_excel(output_file, index=False)
            print(f"\nRULER results saved to '{output_file}'")
    else:
        print("\nNo RULER JSON files found")
    
    # Process other results (if any)
    if other_files:
        df_other = process_dataset(other_files, 'other')
        if df_other is not None:
            output_file = 'other_scores.xlsx'
            df_other.to_excel(output_file, index=False)
            print(f"\nOther results saved to '{output_file}'")
