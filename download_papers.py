from scholarly import scholarly
import requests
from bs4 import BeautifulSoup
import os

# Function to search for papers in a given category
def search_papers(category, num_papers=1):
    search_query = scholarly.search_pubs(category)
    papers = []
    
    for _ in range(num_papers):
        try:
            paper = next(search_query)
            papers.append({
                'title': paper['bib']['title'],
                'url': paper.get('pub_url', '')
            })
        except StopIteration:
            break
            
    return papers

# Function to download paper from Sci-Hub
def download_from_scihub(doi, title):
    if not doi:
        print(f"No DOI found for {title}. Skipping download.")
        return
    
    sci_hub_url = 'https://sci-hub.se/' + doi
    response = requests.get(sci_hub_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    embed = soup.find('embed')
    if embed and embed.get('src'):
        pdf_url = embed['src']
        if pdf_url.startswith('//'):
            pdf_url = 'https:' + pdf_url
        elif pdf_url.startswith('/'):
            pdf_url = 'https://sci-hub.se' + pdf_url
        else:
            pdf_url = pdf_url
        
        pdf_response = requests.get(pdf_url)
        
        file_path = f"{title}.pdf".replace('/', '-')
        with open(file_path, 'wb') as f:
            f.write(pdf_response.content)
            
        print(f"Downloaded: {file_path}")
    else:
        print(f"Paper not found on Sci-Hub: {title}")

# Main function to search and download papers
def main(category):
    papers = search_papers(category)
    for paper in papers:
        print(f"Searching for: {paper['title']}")
        # Assuming the paper URL is a DOI
        download_from_scihub(paper['url'], paper['title'])

if __name__ == "__main__":
    category = 'Deep Learning'
    main(category)
