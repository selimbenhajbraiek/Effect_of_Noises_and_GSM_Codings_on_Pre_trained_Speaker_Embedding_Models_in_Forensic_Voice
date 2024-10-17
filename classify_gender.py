import pandas as pd
import glob

input_files = glob.glob('*.txt')  
for input_text_file in input_files:
    print(f"Processing file: {input_text_file}")
    
    males = pd.DataFrame()
    females = pd.DataFrame()

    chunk_size = 10000  
    total_rows_processed = 0  

    with open(input_text_file, 'r') as file:
        header = file.readline().strip().split(',')

        for chunk in pd.read_csv(file, header=None, names=header, chunksize=chunk_size):
            male_chunk = chunk[(chunk['sex1'].str.lower() == 'male') & (chunk['sex2'].str.lower() == 'male')]
            female_chunk = chunk[(chunk['sex1'].str.lower() == 'female') & (chunk['sex2'].str.lower() == 'female')]

            males = pd.concat([males, male_chunk], ignore_index=True)
            females = pd.concat([females, female_chunk], ignore_index=True)

            total_rows_processed += len(chunk)

            print(f"Processed chunk with {len(chunk)} rows. Total rows processed: {total_rows_processed}. Current male count: {len(males)}, Current female count: {len(females)}.")

    output_base_name = input_text_file.rsplit('.', 1)[0]  
    males_output_file = f"{output_base_name}_males.csv"
    females_output_file = f"{output_base_name}_females.csv"


    males.to_csv(males_output_file, index=False)
    females.to_csv(females_output_file, index=False)

    print(f"Classification complete for '{input_text_file}'. Check '{males_output_file}' and '{females_output_file}'.")

print("All files processed.")
