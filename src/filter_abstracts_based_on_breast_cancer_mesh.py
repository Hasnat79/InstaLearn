import pandas as pd
import argparse
from tqdm import tqdm
# Set up argument parser
parser = argparse.ArgumentParser(description='Filter abstracts based on cardio MeSH qualifiers.')
parser.add_argument('--input_file', type=str, help='Path to the input CSV file')
parser.add_argument('--output_file', type=str, help='Path to the output CSV file')

# Parse the arguments
args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file

# Load the CSV file
df = pd.read_csv(input_file)
# print(f"total rows: {len(df)}")
# exit()
# print(df.describe())



mesh_descriptor_terms_breast_cancer_cat = {
  "Breast Neoplasms": [
    "Carcinoma, Ductal, Breast",
    "Carcinoma, Lobular",
    "Carcinoma, Medullary",
    "Carcinoma, Mucinous",
    "Carcinoma, Papillary",
    "Carcinoma, Signet Ring Cell",
    "Carcinoma, Tubular",
    "Fibroadenoma",
    "Phyllodes Tumor",
    "Inflammatory Breast Neoplasms",
    "Male Breast Neoplasms",
    "Paget's Disease, Mammary",
    "Triple Negative Breast Neoplasms",
    "Her2/neu Positive Breast Neoplasms",
    "Receptor-Positive Breast Neoplasms",
    "Receptor-Negative Breast Neoplasms",
    "Neoplasm Metastasis",
    "Neoplasm Recurrence, Local",
    "Neoplasm, Residual"
  ],
  "Triple Negative Breast Neoplasms": [],
  "Carcinoma, Ductal, Breast": [],
  "Carcinoma, Lobular": [],
  "Carcinoma, Medullary": [],
  "Carcinoma, Mucinous": [],
  "Carcinoma, Papillary": [],
  "Carcinoma, Signet Ring Cell": [],
  "Carcinoma, Tubular": [],
  "Fibroadenoma": [],
  "Phyllodes Tumor": [],
  "Inflammatory Breast Neoplasms": [],
  "Male Breast Neoplasms": [],
  "Paget's Disease, Mammary": [],
  "Her2/neu Positive Breast Neoplasms": [],
  "Receptor-Positive Breast Neoplasms": [],
  "Receptor-Negative Breast Neoplasms": [],
  "Neoplasm Metastasis": [],
  "Neoplasm Recurrence, Local": [],
  "Neoplasm, Residual": []
}

    



# print(df.columns)

# Initialize a dictionary to store the counts
qualifier_counts = {term: 0 for term in mesh_descriptor_terms_breast_cancer_cat.keys()}

# Iterate over each row in the DataFrame
filtered_rows = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    if pd.notna(row['Abstract']):
        for category, sub_categories in mesh_descriptor_terms_breast_cancer_cat.items():
            try:
                if any(sub_cat in row['MeSH_Descriptors'] for sub_cat in sub_categories) or category in row['MeSH_Descriptors']:
                    qualifier_counts[category] += 1
                    row['category'] = category
                    row['sub_category'] = [sub_cat for sub_cat in sub_categories if sub_cat in row['MeSH_Descriptors']]
                    filtered_rows.append(row)
                    break
            except Exception as e:
                print(e)
                print(row['MeSH_Descriptors'])
                print(sub_categories)
                print()
                
            
        

filtered_df = pd.DataFrame(filtered_rows)
print(filtered_df.describe())
# print(filtered_df.columns)

# Total articles with cardio diseases
total_articles = sum(qualifier_counts.values())
print(f"Total articles with breast cancer: {total_articles}")

# Print the counts and examples
for term, count in qualifier_counts.items():
    print(f"{term}: {count}")
print()
print()
#print abstracts for each term
# for term, count in qualifier_counts.items():
#     if qualifier_examples[term]:
#         print(f"Example abstract for {term}: {qualifier_examples[term]}")
#     print()

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv(output_file, index=False)
