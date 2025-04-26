import os
import json
from typing import List

def load_model_outputs(data_dir: str) -> List[str]:
    """
    Loads all JSON or JSONL files in the given directory and extracts the 'model_output_text' field.

    Args:
        data_dir (str): Path to the data directory.

    Returns:
        List[str]: A list of extracted model output texts.
    """
    model_outputs = []

    # Loop through files in the directory
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        if filename.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for record in data:
                        text = record.get('model_output_text')
                        if text:
                            model_outputs.append(text)
                elif isinstance(data, dict):
                    text = data.get('model_output_text')
                    if text:
                        model_outputs.append(text)

        elif filename.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        record = json.loads(line)
                        text = record.get('model_output_text')
                        if text:
                            model_outputs.append(text)

    return model_outputs

if __name__ == "__main__":
    # Example usage
    data_directory = "data/"  # Adjust if needed
    outputs = load_model_outputs(data_directory)
    print(f"Loaded {len(outputs)} model outputs.")
    print(outputs[:3])  # Print first few to check
