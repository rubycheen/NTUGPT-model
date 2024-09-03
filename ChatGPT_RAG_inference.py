import openai

import argparse
import pandas as pd

# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever

openai.api_key = ""
web_page = pd.read_csv('document/clean_content_4.csv')

def ask_chatgpt(prompt):
    
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      temperature=0,
      messages=[
            {"role": "system", "content": "You are an artificial intelligence assistant and National Taiwan University campus guide"},
            {"role": "user", "content": prompt},
        ]
    )
    return completion


def RAG(query):
    embeddings = HuggingFaceBgeEmbeddings(model_name = "BAAI/bge-large-zh-v1.5")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    # relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.55)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[redundant_filter]
    )

    db = FAISS.load_local('embeddings/all_bge_large_chatgpt', embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)


    retrv_urls, retrv_origin_docs, retrv_expand_docs = [], [], []

    retrieve_docs = compression_retriever.get_relevant_documents(query)
    retrv_origin_docs.append([doc.page_content for doc in retrieve_docs])
    url_set = set([doc.metadata['url'] for doc in retrieve_docs])
    retrv_urls.append(url_set)

    contents = {}
    for url in url_set:
        for idx, u in enumerate(web_page['url']):
            if url==u:
                contents[url] = web_page['content'][idx]

    paragraphs = []
    for i in range(len(retrieve_docs)):
        snippet = retrieve_docs[i].page_content
        str_idx = contents[retrieve_docs[i].metadata['url']].find(snippet)
        if str_idx==-1:
            paragraphs.append(snippet)
        else:
            paragraphs.append(contents[retrieve_docs[i].metadata['url']][str_idx:str_idx+128])
    retrv_expand_docs.append(paragraphs)

    paragraph = [f'\n文檔{i+1}:'+paragraphs[i]+'' for i in range(len(paragraphs))]

    prompt = '以下是參考資料，請忽略不相關的文件，回答盡量簡短精要，切勿重複輸出一樣文句子:{}\n請問:{}'.format(','.join(paragraph), query)
    # print(prompt)

    response = ask_chatgpt(prompt)['choices'][0]['message']['content']


    return retrieve_docs, response

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--query', dest='query', type=str, help='Add query')
    args = parser.parse_args()

    retrieve_docs, response = RAG(args.query)

    print("回覆:")
    print(response)

    print(f'參考網頁：')
    for doc in retrieve_docs:
        print(doc.metadata['url'])