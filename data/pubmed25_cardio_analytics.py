import pandas as pd
import os

input = "/scratch/group/instalearn/InstaLearn/data/pubmed25_cardio.csv"

# Load the CSV file
df = pd.read_csv(input)

# Print the columns
print("Columns:", df.columns)

# Print the description of the dataframe
print("Description:\n", df.describe())

# print total row counts
print("Total rows:", len(df))

# Columns: Index(['PMID', 'PMID_Version', 'Title', 'Status', 'IndexingMethod', 'Owner',
#        'Abstract', 'Journal_Title', 'ISSN', 'ISSN_Type', 'Volume', 'Issue',
#        'CitedMedium', 'PubDate_Year', 'PubDate_Month', 'PubDate_Day',
#        'Pagination', 'Authors', 'PublicationTypes', 'Chemicals',
#        'MeSHHeadings_Full', 'MeSH_Descriptors', 'MeSH_Qualifiers',
#        'MeSH_MajorTopics', 'Grants', 'DateCompleted', 'DateRevised',
#        'PublicationHistory', 'ArticleIDs', 'category', 'sub_category'],
#       dtype='object')
# Description:
#                 PMID  PMID_Version   PubDate_Year   PubDate_Day
# count  1.462030e+05      146203.0  135466.000000  17774.000000
# mean   4.493536e+06           1.0    1987.549717     12.238213
# std    2.807417e+06           0.0       5.779336      9.415661
# min    1.570000e+02           1.0    1949.000000      1.000000
# 25%    2.083068e+06           1.0    1984.000000      1.000000
# 50%    3.630927e+06           1.0    1989.000000     14.500000
# 75%    7.402727e+06           1.0    1992.000000     18.000000
# max    1.542699e+07           1.0    1998.000000     31.000000
# Total rows: 146203

# Check for missing values
print("Missing values:\n", df.isnull().sum())
print("=====================================")
# Check the distribution of the 'PubDate_Year' column
print("Publication Year Distribution:\n", df['PubDate_Year'].value_counts().sort_index())
print("=====================================")
# Check the distribution of the 'Status' column
print("Status Distribution:\n", df['Status'].value_counts())
print("=====================================")
# Check the distribution of the 'Journal_Title' column
print("Top 10 Journals:\n", df['Journal_Title'].value_counts().head(10))
print("=====================================")
# Check the distribution of the 'Authors' column
print("Top 10 Authors:\n", df['Authors'].value_counts().head(10))
print("=====================================")
# Check the distribution of the 'category' column
print("Category Distribution:\n", df['category'].value_counts())
print("=====================================")
# Check the distribution of the 'sub_category' column
print("Sub-category Distribution:\n", df['sub_category'].value_counts())
print("=====================================")
# Check the distribution of the 'MeSHHeadings_Full' column
print("Top 10 MeSH Headings:\n", df['MeSHHeadings_Full'].value_counts().head(10))
print("=====================================")

import matplotlib.pyplot as plt

# Create plots directory if it doesn't exist
plots_dir = "/scratch/group/instalearn/InstaLearn/data/plots"
os.makedirs(plots_dir, exist_ok=True)
# Plot the distribution of the 'PubDate_Year' column
plt.figure(figsize=(10, 6))
df['PubDate_Year'].dropna().astype(int).value_counts().sort_index().plot(kind='bar')
plt.title('Publication Year Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'pubdate_year_distribution.png'))
plt.close()

# Plot the distribution of the 'Status' column
plt.figure(figsize=(10, 6))
df['Status'].value_counts().plot(kind='bar')
plt.title('Status Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Status', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14, fontweight='bold')
plt.xticks(rotation=15, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'status_distribution.png'))
plt.close()

# Plot the distribution of the 'Journal_Title' column
plt.figure(figsize=(14, 8))  # Increase the figure size
df['Journal_Title'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Journals', fontsize=16, fontweight='bold')
plt.xlabel('Journal Title', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')  # Rotate x-axis labels and align them to the right
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()  # Adjust layout to fit everything
plt.savefig(os.path.join(plots_dir, 'top_10_journals.png'))
plt.close()

# Plot the distribution of the 'Authors' column
plt.figure(figsize=(14, 8))  # Increase the figure size
df['Authors'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Authors', fontsize=16, fontweight='bold')
plt.xlabel('Authors', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')  # Rotate x-axis labels and align them to the right
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()  # Adjust layout to fit everything
plt.savefig(os.path.join(plots_dir, 'top_10_authors.png'))
plt.close()

# Plot the distribution of the 'category' column
plt.figure(figsize=(10, 6))
df['category'].value_counts().plot(kind='bar')
plt.title('Category Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Category', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14, fontweight='bold')
plt.xticks(rotation=15, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'category_distribution.png'))
plt.close()

# Plot the distribution of the 'sub_category' column
plt.figure(figsize=(14, 8))  # Increase the figure size
df['sub_category'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Sub-category Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Sub-category', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')  # Rotate x-axis labels and align them to the right
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()  # Adjust layout to fit everything
plt.savefig(os.path.join(plots_dir, 'top_10_sub_category_distribution.png'))
plt.close()
