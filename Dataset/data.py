import os
import csv
import json

# Function to write data to a CSV file
def write_to_csv(output_csv_file, formatted_data, is_first_row=False):
    mode = 'w' if is_first_row else 'a'
    with open(output_csv_file, mode, newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        if is_first_row:
            csv_writer.writerow(['text'])  # Set the column name as 'text'
        csv_writer.writerow([formatted_data])

def main():
    train_folder = r'D:\Research Papers\methods2test\methods2test\corpus\json\test\test'
    output_folder = r'D:\Fyp data\TestData'

    # Iterate through all folders in the train_folder directory
    for foldername in os.listdir(train_folder):
        folder_path = os.path.join(train_folder, foldername)
        if os.path.isdir(folder_path):
            # Define the output CSV file path for the current folder
            output_csv_file = os.path.join(output_folder, f'{foldername}.csv')

            # Reset the rows_written counter for each folder
            rows_written = 0

            # Iterate through all files in the current folder
            for filename in os.listdir(folder_path):
                if filename.endswith('_corpus.json'):
                    input_file = os.path.join(folder_path, filename)
                    print(input_file)
                    print()

                    with open(input_file, 'r') as f:
                        input_data = json.load(f)
                        input_text = input_data.get("src_fm_fc_ms_ff", "")  # Change to "src_fm_fc_ms_ff"
                        target_text = input_data.get("target", "")  # Change to "target"

                    formatted_data = f"<s> <<SYS>>Genrate Unit Test Case for this Unit <</SYS>>[INST] {input_text} [/INST] {target_text} </s>"

                    # Write data to the CSV file
                    write_to_csv(output_csv_file, formatted_data, rows_written == 0)  # Set column name in the first row
                    rows_written += 1

if __name__ == "__main__":
    main()
