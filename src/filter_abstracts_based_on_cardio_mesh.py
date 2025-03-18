import pandas as pd
import argparse

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
print(df.describe())

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

mesh_descriptor_terms_cardio_diseases = [
    "Cerebrovascular Disorders",
    "Myocardial Ischemia",
    "Cardiomyopathies",
    "Heart Failure",
    "Arrhythmias, Cardiac",
    "Heart Valve Diseases",
    "Heart Defects, Congenital",
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
    "Cerebrovascular Disorders",
    "Acute Coronary Syndrome",
    "Angina Pectoris",
    "Coronary Disease",
    "Kounis Syndrome",
    "Myocardial Infarction",
    "Myocardial Reperfusion Injury",
    "Myocardial Ischemia",
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
    "Heart Failure, Systolic",
    "Cardiomyopathies",
    "Heart Failure",
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
    "Ventricular Flutter",
    "Arrhythmias, Cardiac",
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
    "Tricuspid Valve Stenosis",
    "Heart Valve Diseases",
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
    "Wolff-Parkinson-White Syndrome",
    "Heart Defects, Congenital"
]
print(df.columns)

# Initialize a dictionary to store the counts
qualifier_counts = {term: 0 for term in mesh_descriptor_terms_cardio_diseases}
qualifier_examples = {term: None for term in mesh_descriptor_terms_cardio_diseases}

# Iterate over each row in the DataFrame
filtered_rows = []

for index, row in df.iterrows():
    if pd.notna(row['Abstract']):
        for term in mesh_descriptor_terms_cardio_diseases:
            if term in row['MeSH_Descriptors']:
                qualifier_counts[term] += 1
                if qualifier_examples[term] is None:
                    qualifier_examples[term] = row['Abstract']
                filtered_rows.append(row)
                break

filtered_df = pd.DataFrame(filtered_rows)
print(filtered_df.describe())
print(filtered_df.columns)

# Total articles with cardio diseases
total_articles = sum(qualifier_counts.values())
print(f"Total articles with cardio diseases: {total_articles}")

# Print the counts and examples
for term, count in qualifier_counts.items():
    print(f"{term}: {count}")
print()
#print abstracts for each term
# for term, count in qualifier_counts.items():
#     if qualifier_examples[term]:
#         print(f"Example abstract for {term}: {qualifier_examples[term]}")
#     print()

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv(output_file, index=False)
