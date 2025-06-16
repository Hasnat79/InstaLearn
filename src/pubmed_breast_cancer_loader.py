from datasets import load_dataset


class PubMedBreastCancerLoader:
    def __init__(self, dataset_name: str):
        self.dataset = load_dataset(dataset_name)
        # print(self.dataset)
        print(f"Loaded dataset: {dataset_name}")

    def __len__(self):
        """Returns the total number of items in the dataset."""
        return sum(len(self.dataset[split]) for split in self.dataset)


if __name__ == "__main__":
    dataset_name = "InstaLearn/pubmed25_breast_cancer"
    data_path = "/scratch/group/instalearn/InstaLearn/data/pubmed25_cardio.csv"

    pubmed_cardio_loader = PubMedBreastCancerLoader(dataset_name)
    print(pubmed_cardio_loader.dataset)
    for item in pubmed_cardio_loader.dataset:
        print(item["PMID"])
        print(item["Abstract"])

