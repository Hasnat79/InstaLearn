import xml.etree.ElementTree as ET
# import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

def parse_pubmed_xml(file_path, article_index=0, parse_all=False, debug=False):
    """
    Parse PubMed XML file and extract attributes of articles.
    
    Args:
        file_path (str): Path to the PubMed XML file
        article_index (int): Index of the article to parse (0-based)
        parse_all (bool): If True, parse all articles in the file
        debug (bool): If True, enable debugging mode
        
    Returns:
        dict or list: Dictionary containing the attributes of a single article or
                     a list of dictionaries for all articles
    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Get all PubmedArticle elements
    articles = root.findall(".//PubmedArticle")
    
    if not articles:
        return {"error": "No PubmedArticle elements found"}
    
    if parse_all:
        all_articles_info = []
        for i, article in enumerate(tqdm(articles)):
            if debug:
                input(f"Press Enter to parse article {i+1} of {len(articles)} (PMID: {article.find('.//PMID').text if article.find('.//PMID') is not None else 'Unknown'})...")
            
            article_info = _extract_article_info(article)
            all_articles_info.append(article_info)
        
        return all_articles_info
    else:
        # Get the article at the specified index
        if article_index < 0 or article_index >= len(articles):
            return {"error": f"Article index {article_index} out of range (0-{len(articles)-1})"}
        
        article = articles[article_index]
        return _extract_article_info(article)

def _extract_article_info(article):
    """
    Extract information from a single PubmedArticle element.
    
    Args:
        article (Element): PubmedArticle Element
        
    Returns:
        dict: Dictionary containing the article's attributes
    """
    # Extract basic article information
    article_info = {}
    
    # PMID
    pmid = article.find(".//PMID")
    if pmid is not None:
        article_info["PMID"] = pmid.text
        article_info["PMID_Version"] = pmid.get("Version")
    
    # Article title
    title = article.find(".//ArticleTitle")
    if title is not None:
        article_info["Title"] = title.text
        
    # Abstract and Abstract Title
    abstract = article.find(".//Abstract")
    if abstract is not None:
        abstract_texts = abstract.findall(".//AbstractText")
        if abstract_texts:
            abstract_content = []
            for abstract_text in abstract_texts:
                # Check if there's a label attribute (often used as a title/section header)
                label = abstract_text.get("Label")
                if label:
                    abstract_content.append(f"**{label}**: {abstract_text.text}")
                else:
                    abstract_content.append(abstract_text.text)
            
            article_info["Abstract"] = "\n\n".join([text for text in abstract_content if text])
    
    # Journal information
    journal = article.find(".//Journal")
    if journal is not None:
        journal_info = {}
        
        # Journal title
        journal_title = journal.find(".//Title")
        if journal_title is not None:
            journal_info["Title"] = journal_title.text
        
        # ISSN
        issn = journal.find(".//ISSN")
        if issn is not None:
            journal_info["ISSN"] = issn.text
            journal_info["ISSN_Type"] = issn.get("IssnType")
        
        # Journal issue
        issue = journal.find(".//JournalIssue")
        if issue is not None:
            journal_info["Volume"] = issue.find(".//Volume").text if issue.find(".//Volume") is not None else None
            journal_info["Issue"] = issue.find(".//Issue").text if issue.find(".//Issue") is not None else None
            journal_info["CitedMedium"] = issue.get("CitedMedium")
            
            # Publication date
            pub_date = issue.find(".//PubDate")
            if pub_date is not None:
                date_parts = {}
                for part in pub_date:
                    date_parts[part.tag] = part.text
                journal_info["PubDate"] = date_parts
        
        article_info["Journal"] = journal_info
    
    # Authors
    authors = article.findall(".//Author")
    if authors:
        author_list = []
        for author in authors:
            author_info = {}
            for elem in author:
                author_info[elem.tag] = elem.text
            author_info["ValidYN"] = author.get("ValidYN")
            author_list.append(author_info)
        article_info["Authors"] = author_list
    
    # Pagination
    pagination = article.find(".//Pagination/MedlinePgn")
    if pagination is not None:
        article_info["Pagination"] = pagination.text
    
    # Publication types
    pub_types = article.findall(".//PublicationType")
    if pub_types:
        article_info["PublicationTypes"] = [{"Type": pt.text, "UI": pt.get("UI")} for pt in pub_types]
    
    # Chemicals
    chemicals = article.findall(".//Chemical")
    if chemicals:
        chemical_list = []
        for chemical in chemicals:
            chem_info = {}
            registry = chemical.find(".//RegistryNumber")
            if registry is not None:
                chem_info["RegistryNumber"] = registry.text
            
            substance = chemical.find(".//NameOfSubstance")
            if substance is not None:
                chem_info["Name"] = substance.text
                chem_info["UI"] = substance.get("UI")
            
            chemical_list.append(chem_info)
        
        article_info["Chemicals"] = chemical_list
    
    # MeSH headings
    mesh_headings = article.findall(".//MeshHeading")
    if mesh_headings:
        mesh_list = []
        for heading in mesh_headings:
            mesh_info = {}
            
            descriptor = heading.find(".//DescriptorName")
            if descriptor is not None:
                mesh_info["Descriptor"] = descriptor.text
                mesh_info["DescriptorUI"] = descriptor.get("UI")
                mesh_info["DescriptorMajorTopic"] = descriptor.get("MajorTopicYN")
            
            qualifiers = heading.findall(".//QualifierName")
            if qualifiers:
                mesh_info["Qualifiers"] = [
                    {
                        "Name": q.text,
                        "UI": q.get("UI"),
                        "MajorTopicYN": q.get("MajorTopicYN")
                    } for q in qualifiers
                ]
            
            mesh_list.append(mesh_info)
        
        article_info["MeSHHeadings"] = mesh_list
    
    # Grants
    grants = article.findall(".//Grant")
    if grants:
        grant_list = []
        for grant in grants:
            grant_info = {}
            for elem in grant:
                grant_info[elem.tag] = elem.text
            grant_list.append(grant_info)
        
        article_info["Grants"] = grant_list
    
    # Completion and revision dates
    date_completed = article.find(".//DateCompleted")
    if date_completed is not None:
        completed = {}
        for elem in date_completed:
            completed[elem.tag] = elem.text
        article_info["DateCompleted"] = completed
    
    date_revised = article.find(".//DateRevised")
    if date_revised is not None:
        revised = {}
        for elem in date_revised:
            revised[elem.tag] = elem.text
        article_info["DateRevised"] = revised
    
    # Other metadata
    citation_status = article.find(".//MedlineCitation")
    if citation_status is not None:
        article_info["Status"] = citation_status.get("Status")
        article_info["IndexingMethod"] = citation_status.get("IndexingMethod")
        article_info["Owner"] = citation_status.get("Owner")
    
    # Publication status and history
    pub_status = article.find(".//PublicationStatus")
    if pub_status is not None:
        article_info["PublicationStatus"] = pub_status.text
    
    history = article.findall(".//PubMedPubDate")
    if history:
        history_info = []
        for date in history:
            date_info = {"PubStatus": date.get("PubStatus")}
            for elem in date:
                date_info[elem.tag] = elem.text
            history_info.append(date_info)
        article_info["History"] = history_info
    
    # Article IDs
    article_ids = article.findall(".//ArticleId")
    if article_ids:
        id_info = {}
        for aid in article_ids:
            id_info[aid.get("IdType")] = aid.text
        article_info["ArticleIDs"] = id_info
    
    return article_info

def prepare_article_for_csv(article_info):
    """
    Prepare article information for CSV format
    
    Args:
        article_info (dict): Dictionary containing article attributes
        
    Returns:
        dict: Flattened dictionary suitable for CSV export
    """
    csv_data = {}
    
    # Basic information
    csv_data['PMID'] = article_info.get('PMID', '')
    csv_data['PMID_Version'] = article_info.get('PMID_Version', '')
    csv_data['Title'] = article_info.get('Title', '')
    csv_data['Status'] = article_info.get('Status', '')
    csv_data['IndexingMethod'] = article_info.get('IndexingMethod', '')
    csv_data['Owner'] = article_info.get('Owner', '')
    
    # Abstract
    csv_data['Abstract'] = article_info.get('Abstract', '').replace('\n', ' ').replace('"', '""')
    
    # Journal information
    journal = article_info.get('Journal', {})
    csv_data['Journal_Title'] = journal.get('Title', '')
    csv_data['ISSN'] = journal.get('ISSN', '')
    csv_data['ISSN_Type'] = journal.get('ISSN_Type', '')
    csv_data['Volume'] = journal.get('Volume', '')
    csv_data['Issue'] = journal.get('Issue', '')
    csv_data['CitedMedium'] = journal.get('CitedMedium', '')
    
    # Publication date
    pub_date = journal.get('PubDate', {})
    csv_data['PubDate_Year'] = pub_date.get('Year', '')
    csv_data['PubDate_Month'] = pub_date.get('Month', '')
    csv_data['PubDate_Day'] = pub_date.get('Day', '')
    
    # Pagination
    csv_data['Pagination'] = article_info.get('Pagination', '')
    
    # Authors - combine into a single field
    authors = article_info.get('Authors', [])
    author_strings = []
    for author in authors:
        name_parts = []
        if author.get('ForeName'):
            name_parts.append(author.get('ForeName'))
        if author.get('LastName'):
            name_parts.append(author.get('LastName'))
        
        if name_parts:
            author_string = f"{' '.join(name_parts)} ({author.get('Initials', '')})"
            author_strings.append(author_string)
    
    csv_data['Authors'] = '; '.join(author_strings)
    
    # Publication types - combine into a single field
    pub_types = article_info.get('PublicationTypes', [])
    pub_type_strings = [f"{pt.get('Type', '')} ({pt.get('UI', '')})" for pt in pub_types]
    csv_data['PublicationTypes'] = '; '.join(pub_type_strings)
    
    # Chemicals - combine into a single field
    chemicals = article_info.get('Chemicals', [])
    chemical_strings = [f"{chem.get('Name', '')} (Registry: {chem.get('RegistryNumber', '')}, UI: {chem.get('UI', '')})" for chem in chemicals]
    csv_data['Chemicals'] = '; '.join(chemical_strings)
    
    # MeSH headings - Handle both as combined field and individual columns
    mesh = article_info.get('MeSHHeadings', [])
    
    # Combined field for all MeSH headings
    mesh_strings = []
    for heading in mesh:
        major = "Yes" if heading.get('DescriptorMajorTopic') == "Y" else "No"
        mesh_string = f"{heading.get('Descriptor', '')} (UI: {heading.get('DescriptorUI', '')}, Major: {major})"
        
        qualifiers = heading.get('Qualifiers', [])
        if qualifiers:
            qualifier_strings = []
            for q in qualifiers:
                q_major = "Yes" if q.get('MajorTopicYN') == "Y" else "No"
                qualifier_strings.append(f"{q.get('Name', '')} (UI: {q.get('UI', '')}, Major: {q_major})")
            
            mesh_string += " - Qualifiers: " + "; ".join(qualifier_strings)
        
        mesh_strings.append(mesh_string)
    
    csv_data['MeSHHeadings_Full'] = '; '.join(mesh_strings)
    
    # Individual MeSH columns - just descriptor name and whether it's a major topic
    mesh_descriptors = []
    mesh_qualifiers = []
    mesh_major_topics = []
    
    for heading in mesh:
        descriptor = heading.get('Descriptor', '')
        is_major = "1" if heading.get('DescriptorMajorTopic') == "Y" else "0"
        
        mesh_descriptors.append(descriptor)
        mesh_major_topics.append(is_major)
        
        # Add qualifiers
        qualifiers = heading.get('Qualifiers', [])
        qualifier_strings = []
        for q in qualifiers:
            q_name = q.get('Name', '')
            q_major = "1" if q.get('MajorTopicYN') == "Y" else "0"
            qualifier_strings.append(f"{q_name} ({q_major})")
        
        if qualifier_strings:
            mesh_qualifiers.append("; ".join(qualifier_strings))
        else:
            mesh_qualifiers.append("")
    
    # Add MeSH columns
    csv_data['MeSH_Descriptors'] = '|'.join(mesh_descriptors)
    csv_data['MeSH_Qualifiers'] = '|'.join(mesh_qualifiers)
    csv_data['MeSH_MajorTopics'] = '|'.join(mesh_major_topics)
    
    # Grants - combine into a single field
    grants = article_info.get('Grants', [])
    grant_strings = []
    for grant in grants:
        grant_parts = [f"ID: {grant.get('GrantID', '')}"]
        if grant.get('Acronym'):
            grant_parts.append(f"Acronym: {grant.get('Acronym')}")
        if grant.get('Agency'):
            grant_parts.append(f"Agency: {grant.get('Agency')}")
        if grant.get('Country'):
            grant_parts.append(f"Country: {grant.get('Country')}")
        
        grant_strings.append(", ".join(grant_parts))
    
    csv_data['Grants'] = '; '.join(grant_strings)
    
    # Dates
    date_completed = article_info.get('DateCompleted', {})
    if date_completed:
        csv_data['DateCompleted'] = f"{date_completed.get('Year', '')}-{date_completed.get('Month', '')}-{date_completed.get('Day', '')}"
    else:
        csv_data['DateCompleted'] = ''
    
    date_revised = article_info.get('DateRevised', {})
    if date_revised:
        csv_data['DateRevised'] = f"{date_revised.get('Year', '')}-{date_revised.get('Month', '')}-{date_revised.get('Day', '')}"
    else:
        csv_data['DateRevised'] = ''
    
    # Publication history - combine into a single field
    history = article_info.get('History', [])
    history_strings = []
    for entry in history:
        status = entry.get('PubStatus', '')
        date_parts = []
        if entry.get('Year'):
            date_parts.append(entry.get('Year'))
        if entry.get('Month'):
            date_parts.append(entry.get('Month'))
        if entry.get('Day'):
            date_parts.append(entry.get('Day'))
        
        date_str = "-".join(date_parts)
        
        time_parts = []
        if entry.get('Hour'):
            time_parts.append(entry.get('Hour'))
        if entry.get('Minute'):
            time_parts.append(entry.get('Minute'))
        
        if time_parts:
            time_str = ":".join(time_parts)
            date_str += f" {time_str}"
        
        history_strings.append(f"{status}: {date_str}")
    
    csv_data['PublicationHistory'] = '; '.join(history_strings)
    
    # Article IDs - combine into a single field
    article_ids = article_info.get('ArticleIDs', {})
    id_strings = [f"{id_type}: {id_value}" for id_type, id_value in article_ids.items()]
    csv_data['ArticleIDs'] = '; '.join(id_strings)
    
    return csv_data

def get_csv_headers(articles_info):
    """
    Get the CSV headers from a list of article dictionaries
    
    Args:
        articles_info (list): List of article dictionaries
        
    Returns:
        list: List of CSV headers
    """
    if not articles_info:
        return []
    
    # Use the first article to determine headers
    if isinstance(articles_info, list):
        first_article = prepare_article_for_csv(articles_info[0])
    else:
        first_article = prepare_article_for_csv(articles_info)
    
    return list(first_article.keys())

def write_csv_file(file_path, articles_info):
    """
    Write article information to a CSV file
    
    Args:
        file_path (str): Path to the output CSV file
        articles_info (list or dict): List of article dictionaries or a single article dictionary
        
    Returns:
        bool: True if successful, False otherwise
    """
    import csv
    
    try:
        # Convert to list if it's a single article
        if not isinstance(articles_info, list):
            articles_info = [articles_info]
        
        # Prepare data for CSV
        csv_data = []
        for article in articles_info:
            csv_data.append(prepare_article_for_csv(article))
        
        # Get headers
        headers = get_csv_headers(articles_info)
        
        # Write to CSV file
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(csv_data)
        
        return True
    
    except Exception as e:
        print(f"Error writing to CSV file: {e}")
        return False

def main():
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Parse PubMed XML files')
    parser.add_argument('file_path', help='Path to the PubMed XML file')
    parser.add_argument('--index', type=int, default=0, help='Index of the article to parse (0-based)')
    parser.add_argument('--all', action='store_true', help='Parse all articles in the file')
    parser.add_argument('--debug', action='store_true', help='Enable debugging mode')
    parser.add_argument('--output', help='Output file path (default: pubmed_articles.csv)')
    
    args = parser.parse_args()
    
    # Parse the XML file
    if args.all:
        articles_info = parse_pubmed_xml(args.file_path, parse_all=True, debug=args.debug)
        
        if args.debug:
            # Display summary of each article in debug mode
            for i, article_info in enumerate(articles_info):
                print(f"\n{'='*80}\nArticle {i+1} of {len(articles_info)}\n{'='*80}\n")
                print(f"PMID: {article_info.get('PMID', 'N/A')}")
                print(f"Title: {article_info.get('Title', 'N/A')}")
                print(f"Journal: {article_info.get('Journal', {}).get('Title', 'N/A')}")
                print(f"Publication Date: {article_info.get('Journal', {}).get('PubDate', {}).get('Year', 'N/A')}")
                print(f"Abstract: {article_info.get('Abstract', 'N/A')[:100]}...")
                input("Press Enter to continue...")
        
        # Save all articles to a single CSV file
        output_file = args.output or "pubmed_articles.csv"
        if write_csv_file(output_file, articles_info):
            print(f"\nProcessed {len(articles_info)} articles")
            print(f"Data saved to '{output_file}'")
        else:
            print("\nFailed to write CSV file")
    
    else:
        # Parse a single article
        article_info = parse_pubmed_xml(args.file_path, article_index=args.index, debug=args.debug)
        
        if "error" in article_info:
            print(f"Error: {article_info['error']}")
            return
        
        if args.debug:
            # Display summary in debug mode
            print(f"\n{'='*80}\nArticle Details\n{'='*80}\n")
            print(f"PMID: {article_info.get('PMID', 'N/A')}")
            print(f"Title: {article_info.get('Title', 'N/A')}")
            print(f"Journal: {article_info.get('Journal', {}).get('Title', 'N/A')}")
            print(f"Publication Date: {article_info.get('Journal', {}).get('PubDate', {}).get('Year', 'N/A')}")
            print(f"Abstract: {article_info.get('Abstract', 'N/A')[:100]}...")
        
        # Save to CSV file
        output_file = args.output or "pubmed_article.csv"
        if write_csv_file(output_file, article_info):
            print(f"\nData saved to '{output_file}'")
        else:
            print("\nFailed to write CSV file")

if __name__ == "__main__":
    main()
