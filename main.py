import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import numpy as np

import openai
import chromadb
from uuid import uuid4
import requests

openai.api_key = 'SECRETKEY'
POLYGON_API_KEY = 'SECRETKEY'
HF_API_KEY = 'SECRETKEY'
EXA_API_KEY = 'SECRETKEY'  # Add Exa AI API key

class FinanceNewsEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self, api_key: str = HF_API_KEY, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            import requests
        except ImportError:
            raise ValueError(
                "The requests python package is not installed. Please install it with `pip install requests`"
            )
        self._api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    def __call__(self, texts: Documents) -> Embeddings:
        # Call HuggingFace Embedding API for each document
        embedding =  self._session.post(  # type: ignore
            self._api_url, json={"inputs": texts, "options": {"wait_for_model": True}}
        ).json()
        return embedding

client = chromadb.Client()
collection = client.create_collection(name="stock_sentiment")

def get_stock_news_in_range(stock_ticker: str, date_after:str, date_before:str, debug=False):
    query = f'https://api.polygon.io/v2/reference/news?ticker={stock_ticker}&limit=10&published_utc.gt={date_after}&published_utc.lt={date_before}&sort=published_utc&order=desc&apiKey={POLYGON_API_KEY}'
    if debug:
        print(query)
    return requests.get(query).json()

def get_exa_news(stock_ticker: str, date_after: str, date_before: str):
    url = "https://api.exa.ai/search/news"
    headers = {
        "Authorization": f"Bearer {EXA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": f"{stock_ticker} stock news",
        "num_results": 10,
        "start_published_date": date_after,
        "end_published_date": date_before
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

def add_to_chroma_db(stock_ticker, date_after, date_before):
    poly_outputs = get_stock_news_in_range(stock_ticker, date_after, date_before)
    exa_outputs = get_exa_news(stock_ticker, date_after, date_before)
    
    text_op = ''
    for res in poly_outputs['results']:
        text_op += res['published_utc']
        text_op += ':'
        text_op += '\n\n'
        text_op += res['title']
    
    for res in exa_outputs['results']:
        text_op += res['published_date']
        text_op += ':'
        text_op += '\n\n'
        text_op += res['title']
    
    collection.add(
      documents=[text_op],
      metadatas=[{"source": "Polygon-API and Exa-API"}],
      ids=[str(uuid4())]
    )

def get_text(prompts, top):
  query_result = collection.query(query_texts=prompts, n_results=top)
  response = {}

  index = 0
  
  for id in query_result['ids'][0]:
    response[id] = {}
    response[id]['text'] = query_result['documents'][0][index]
    response[id]['metadata'] = query_result['metadatas'][0][index]
    index += 1

  return response

def analyze_sentiment(data):
  content = data['text']
  source = data['metadata']['source']
  context = "You are an expert investment analyst. Given news regarding an \
  individual stock or a set of stocks, you have experience performing high level \
  sentiment analysis as expected for an investment analyst. \
  This analysis should consider the content of the news article, past stock performance over short term (1-3 months), medium term (5-6 months), and long term (1 year), as well as whether or not\
  the source of the provided article is reputable."
  sentiment_context = "Produce a JSON object containing\
  a confidence score ranging from -1 to +1 with values closer to -1 indicating a strong bearish sentiment, values closer to \
  +1 indicating a strong bullish sentiment, and 0 indicating entirely neutral sentiment. This JSON object should also contain an entry\
  to indicate whether the sentiment was bullish, bearish or neutral. For example your answer should look like: {score: 0.567887, sentiment: bullish} \
  Other than the JSON object there should be no text in your response."
  input = f"The source of the article is {source}. Here is the content: {content}"
  messages = [{'role': 'system', 'content': context}, {'role':'user', 'content': sentiment_context+input}]
  sentiment_score = openai.ChatCompletion.create(model='gpt-4', messages=messages)
  messages.append(sentiment_score['choices'][0]['message'])
  
  analysis_context = "Given the sentiments that you extracted from the articles, provide your understanding of where \
  the security stands right now, how do you project it will perform the short term (3 months) and medium term (6 months),\
  and long term (1yr+)? Provide a buy/sell/hold recommendation. Format and present you answer in a manner which matches that \
  of a seasoned stock analyst while keeping it concise and digestible."
  messages.append({'role':'user', 'content': analysis_context})
  sentiment_analysis = openai.ChatCompletion.create(model='gpt-4', messages=messages)
  score = sentiment_score['choices'][0]['message']['content']
  analysis = sentiment_analysis['choices'][0]['message']['content']
  return score, analysis

st.title("🔎 SIA - Stock Analyzer")
query = st.text_input("Enter a stock ticker to analyze sentiment:", value="TSLA")
add_to_chroma_db('TSLA', '2023-03-01', '2023-07-01')
add_to_chroma_db('RIVN', '2023-03-01', '2023-07-01')
add_to_chroma_db('AAPL', '2023-03-01', '2023-07-01')
if query:
  prompts = [f"What is the current market sentiment for {query}?"]
  add_to_chroma_db(query, '2023-03-01', '2023-07-01')
  data = get_text(prompts, 2)
  for key,value in data.items():
    score, analysis = analyze_sentiment(value)
    
    st.markdown(f"**Sentiment Score:** {score}")
    st.markdown(f"**Sentiment Analysis:** {analysis}")
