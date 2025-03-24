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

qualifier_terms_cardio_diseases = [
    "blood",
    "cerebrospinal fluid",
    "chemically induced",
    "classification",
    "complications",
    "congenital",
    "diagnosis",
    "diagnostic imaging",
    "diet therapy",
    "drug therapy",
    "economics",
    "embryology",
    "enzymology",
    "epidemiology",
    "ethnology",
    "etiology",
    "genetics",
    "history",
    "immunology",
    "metabolism",
    "microbiology",
    "mortality",
    "nursing",
    "parasitology",
    "pathology",
    "physiopathology",
    "prevention & control",
    "psychology",
    "radiotherapy",
    "surgery",
    "therapy",
    "urine",
    "veterinary",
    "virology"
]

mesh_descriptor_terms_cardio_cat = {
    "Cerebrovascular Disorders": [
        "Basal Ganglia Cerebrovascular Disease",
        "Brain Ischemia",
        "Carotid Artery Diseases",
        "Cerebral Small Vessel Diseases",
        "Cerebrovascular Trauma",
        "Dementia, Vascular",
        "Intracranial Arterial Diseases",
        "Intracranial Arteriovenous Malformations",
        "Intracranial Embolism and Thrombosis",
        "Intracranial Hemorrhages",
        "Leukomalacia, Periventricular",
        "Sneddon Syndrome",
        "Stroke",
        "Susac Syndrome",
        "Vascular Headaches",
        "Vasculitis, Central Nervous System",
        "Vasospasm, Intracranial",
        "Cerebrovascular Disorders"
    ],
    "Myocardial Ischemia": [
        "Acute Coronary Syndrome",
        "Angina Pectoris",
        "Coronary Disease",
        "Kounis Syndrome",
        "Myocardial Infarction",
        "Myocardial Reperfusion Injury",
        "Myocardial Ischemia"
    ],
    "Cardiomyopathies": [
        "Arrhythmogenic Right Ventricular Dysplasia",
        "Cardiomyopathy, Alcoholic",
        "Cardiomyopathy, Dilated",
        "Cardiomyopathy, Hypertrophic",
        "Cardiomyopathy, Restrictive",
        "Chagas Cardiomyopathy",
        "Diabetic Cardiomyopathies",
        "Endocardial Fibroelastosis",
        "Endomyocardial Fibrosis",
        "Glycogen Storage Disease Type IIb",
        "Kearns-Sayre Syndrome",
        "Myocardial Reperfusion Injury",
        "Myocarditis",
        "Sarcoglycanopathies",
        "Cardio-Renal Syndrome",
        "Dyspnea, Paroxysmal",
        "Edema, Cardiac",
        "Heart Failure, Diastolic",
        "Heart Failure, Systolic"
    ],
    "Heart Failure": [
        "Cardio-Renal Syndrome",
        "Dyspnea, Paroxysmal",
        "Edema, Cardiac",
        "Heart Failure, Diastolic",
        "Heart Failure, Systolic"
    ],
    "Arrhythmias, Cardiac": [
        "Arrhythmia, Sinus",
        "Atrial Fibrillation",
        "Atrial Flutter",
        "Bradycardia",
        "Brugada Syndrome",
        "Cardiac Complexes, Premature",
        "Commotio Cordis",
        "Heart Block",
        "Long QT Syndrome",
        "Parasystole",
        "Pre-Excitation Syndromes",
        "Tachycardia",
        "Ventricular Fibrillation",
        "Ventricular Flutter"
    ],
    "Heart Valve Diseases": [
        "Aortic Valve Insufficiency",
        "Aortic Valve Stenosis",
        "Heart Valve Prolapse",
        "Mitral Valve Insufficiency",
        "Mitral Valve Stenosis",
        "Pulmonary Atresia",
        "Pulmonary Valve Insufficiency",
        "Pulmonary Valve Stenosis",
        "Tricuspid Atresia",
        "Tricuspid Valve Insufficiency",
        "Tricuspid Valve Stenosis"
    ],
    "Heart Defects, Congenital": [
        "22q11 Deletion Syndrome",
        "Alagille Syndrome",
        "Aortic Coarctation",
        "Arrhythmogenic Right Ventricular Dysplasia",
        "Barth Syndrome",
        "Cor Triatriatum",
        "Coronary Vessel Anomalies",
        "Crisscross Heart",
        "Dextrocardia",
        "Ductus Arteriosus, Patent",
        "Ebstein Anomaly",
        "Ectopia Cordis",
        "Eisenmenger Complex",
        "Heart Septal Defects",
        "Heterotaxy Syndrome",
        "Hypoplastic Left Heart Syndrome",
        "Isolated Noncompaction of the Ventricular Myocardium",
        "LEOPARD Syndrome",
        "Levocardia",
        "Long QT Syndrome",
        "Marfan Syndrome",
        "Noonan Syndrome",
        "Tetralogy of Fallot",
        "Transposition of Great Vessels",
        "Tricuspid Atresia",
        "Trilogy of Fallot",
        "Trisomy 13 Syndrome",
        "Trisomy 18 Syndrome",
        "Turner Syndrome",
        "Wolff-Parkinson-White Syndrome"
    ]
}



# print(df.columns)

# Initialize a dictionary to store the counts
qualifier_counts = {term: 0 for term in mesh_descriptor_terms_cardio_cat.keys()}

# Iterate over each row in the DataFrame
filtered_rows = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    if pd.notna(row['Abstract']):
        for category, sub_categories in mesh_descriptor_terms_cardio_cat.items():
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
print(f"Total articles with cardio diseases: {total_articles}")

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
