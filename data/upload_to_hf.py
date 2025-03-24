from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd




dataset = load_dataset('hasnat79/pubmed25_cardio') 

print(dataset)

# Load the new dataset from the CSV file
new_data_path = "/scratch/group/instalearn/InstaLearn/data/pubmed25_cardio.csv"
new_data = pd.read_csv(new_data_path)

# Convert the pandas DataFrame to a Hugging Face Dataset
new_dataset = Dataset.from_pandas(new_data)

# Update the existing dataset with the new dataset
dataset = DatasetDict({"train": new_dataset})

print(dataset)
dataset.push_to_hub("hasnat79/pubmed25_cardio",token="HF_TOKEN")

