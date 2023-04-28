from langchain.text_splitter import TextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.sitemap import SitemapLoader
import yaml
import os
import nest_asyncio
from bs4 import BeautifulSoup
import requests
import xml.etree.ElementTree as ET

## Main Function: Run on startup ##

if __name__ == "__main__":

    nest_asyncio.apply()

    ## Step 1: Open Config File

    with open('config.yaml', 'r') as config:
        config_file = yaml.safe_load(config)

    ## Step 2: Get necessary api keys

    os.environ['OPENAI_API_KEY'] = config_file["openai_token"]
    embeddings = OpenAIEmbeddings()

    ## Step 3: Use SitemapLoader to scrape webpages

    docs = []

    for site in config_file["web_sites"]:

        sitemap_loader = SitemapLoader(web_path = site)

        sitemap_loader.requests_per_second = 25

        docs = docs + sitemap_loader.load()

    # Clean data

    text_splitter = CharacterTextSplitter(chunk_size=8000, chunk_overlap=3000)

    docs = text_splitter.split_documents(docs)

    ## Final Step: Create vector database with openai embeddings

    db = FAISS.from_documents(docs, embeddings)

    db.save_local("./vectorstore")

   
