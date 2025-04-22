from datasets import load_dataset


class PubMedCardioLoader:
    def __init__(self, dataset_name: str):
        self.dataset = load_dataset(dataset_name)
        # print(self.dataset)
        print(f"Loaded dataset: {dataset_name}")

    def __iter__(self):
        """Returns an iterator over the dataset."""
        for split in self.dataset:
            for item in self.dataset[split]:
                yield item
    def __len__(self):
        """Returns the total number of items in the dataset."""
        return sum(len(self.dataset[split]) for split in self.dataset)


if __name__ == "__main__":
    dataset_name = "hasnat79/pubmed25_cardio"
    data_path = "/scratch/group/instalearn/InstaLearn/data/pubmed25_cardio.csv"

    pubmed_cardio_loader = PubMedCardioLoader(dataset_name)
    for item in pubmed_cardio_loader:
        print(item["PMID"])
        print(item["Abstract"])

#     DatasetDict({
#     train: Dataset({
#         features: ['PMID', 'PMID_Version', 'Title', 'Status', 'IndexingMethod', 'Owner', 'Abstract', 'Journal_Title', 'ISSN', 'ISSN_Type', 'Volume', 'Issue', 'CitedMedium', 'PubDate_Year', 'PubDate_Month', 'PubDate_Day', 'Pagination', 'Authors', 'PublicationTypes', 'Chemicals', 'MeSHHeadings_Full', 'MeSH_Descriptors', 'MeSH_Qualifiers', 'MeSH_MajorTopics', 'Grants', 'DateCompleted', 'DateRevised', 'PublicationHistory', 'ArticleIDs', 'category', 'sub_category'],
#         num_rows: 864244
#     })
# })