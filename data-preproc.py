from langchain.text_splitter import TextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.sitemap import SitemapLoader
import yaml
import os

## Main Function: Run on startup ##

if __name__ == "__main__":

    ## Step 1: Open Config File

    with open('config.yaml', 'r') as config:
        config_file = yaml.safe_load(config)

    ## Step 2: Get necessary api keys

    os.environ['OPENAI_API_KEY'] = config_file["openai_token"]

    ## Step 3: Use SitemapLoader to scrape webpage

    sitemap_loader = SitemapLoader(web_path=config_file["web_site"])

    sitemap_loader.requests_per_second = 50

    docs = sitemap_loader.load()

    ## Step 4: Clean Docs and Split

    preproc_docs = []
    metadatas = [] 

    for doc in docs:
        lines = (line.strip() for line in (doc.page_content).splitlines())
        final = '\n'.join(line for line in lines if line)
    
        final = final.replace("\n", " ")
    
        if "https://based.cooking/tags" not in doc.metadata["source"]:
            preproc_docs.append(final)
            metadatas.append(doc.metadata)
        
    text_splitter = TokenTextSplitter(        
        chunk_size = 1000,
        chunk_overlap  = 200
    )
    
    documents = text_splitter.create_documents(preproc_docs, metadatas=metadatas)

    ## Step 5: Create embeddings and vectorstore

    embedding = OpenAIEmbeddings()

    db = FAISS.from_documents(documents, embedding)

    ## Final Step: Save Vectorstore

    db.save_local(config_file["vectorstore"])
