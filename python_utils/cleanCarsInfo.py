import os
import sys
import re

def extract_model_info(text):
    # This regex captures 4-line blocks with the required tags
    pattern = re.compile(
         r"(Predicted|Actual|Accuracies|Errors)\s*[:=]\s*(\[[^\]]*\]|\d+\.\d+|\d+)",
        re.DOTALL
    )
    blocks = pattern.findall(text)
    return blocks  # Return only the last 4

def process_file(input_path, output_path):
    with open(input_path, 'r') as f:
        text = f.read()

    info = extract_model_info(text)

    
    with open(output_path, 'w') as f:
        for i in info:
            f.write(f"{i[0]}: {i[1]}\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".info"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_file(input_path, output_path)

if __name__ == "__main__":
    main()