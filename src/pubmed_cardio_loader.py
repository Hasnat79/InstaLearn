from datasets import load_dataset


class PubMedCardioLoader:
    def __init__(self, dataset_name: str):
        self.dataset = load_dataset(dataset_name)
        # print(self.dataset)
        print(f"Loaded dataset: {dataset_name}")

    def __len__(self):
        """Returns the total number of items in the dataset."""
        return sum(len(self.dataset[split]) for split in self.dataset)


if __name__ == "__main__":
    dataset_name = "InstaLearn/pubmed25_cardio"
    data_path = "/scratch/group/instalearn/InstaLearn/data/pubmed25_cardio.csv"

    pubmed_cardio_loader = PubMedCardioLoader(dataset_name)
    for item in pubmed_cardio_loader:
        print(item["PMID"])
        print(item["Abstract"])
