import re
from sudachipy import dictionary, tokenizer
from sudachipy.tokenizer import Tokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import stleamlit as st

import numpy as np
#import json
import requests

def extract_text_from_pdf(pdf_path):
    """PDFファイルからテキストを抽出する。
    
    Args:
    - pdf_path (str): PDFファイルへのパス。
    
    Returns:
    - str: PDFから抽出されたテキスト。
    """
    import PyPDF2
    
    file = pdf_path  # pdf_path is now a BytesIO object
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

def create_wordcloud(text, stopwords=None, background_color='white', max_words=200, width=800, height=400):
    """テキストからワードクラウドを生成する。
    
    Args:
    - text (str): 入力テキスト。
    - stopwords (set): ストップワードの集合。
    - background_color (str): 背景色。
    - max_words (int): 最大単語数。
    - width (int): 画像の幅。
    - height (int): 画像の高さ。
    
    Returns:
    - fig: 生成されたワードクラウドのfigureオブジェクト。
    """
    sudachi_tokenizer = dictionary.Dictionary().create()
    text = " ".join([m.surface() for m in sudachi_tokenizer.tokenize(text)])
    wc = WordCloud(font_path='./Arial Unicode.ttf', stopwords=stopwords, background_color=background_color, max_words=max_words, width=width, height=height)
    wc.generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig


def extract_nouns(text):
    """テキストから名詞のみを抽出する。

    Args:
    - text (str): 入力テキスト。
    
    Returns:
    - list: 名詞のリスト。
    """
    sudachi_tokenizer = dictionary.Dictionary().create()
    nouns = [m.surface() for m in sudachi_tokenizer.tokenize(text) if m.part_of_speech()[0] == "名詞"]
    return nouns

# ローカルで動かす場合はjsonで読み込む
# with open('secret.json') as f:
#     secret = json.load(f)

# BASE_URL = secret["COTOHA_BASE_URL"]
# CLIENT_ID = secret["COTOHA_ID"]
# CLIENT_SECRET = secret["COTOHA_SECRET"]

BASE_URL = st.secret["COTOHA_BASE_URL"]
CLIENT_ID = st.secret["COTOHA_ID"]
CLIENT_SECRET = st.secret["COTOHA_SECRET"]

def get_cotoha_acces_token():

    token_url = "https://api.ce-cotoha.com/v1/oauth/accesstokens"

    headers = {
        "Content-Type": "application/json",
        "charset": "UTF-8"
    }

    data = {
        "grantType": "client_credentials",
        "clientId": CLIENT_ID,
        "clientSecret": CLIENT_SECRET
    }

    response = requests.post(token_url,
                        headers=headers,
                        data=json.dumps(data))

    access_token = response.json()["access_token"]

    return access_token


def cotoha_sentiment_analyze(access_token, sentence):
    base_url = BASE_URL
    headers = {
        "Content-Type": "application/json",
        "charset": "UTF-8",
        "Authorization": "Bearer {}".format(access_token)
    }
    data = {
        "sentence": sentence,
    }
    response = requests.post(base_url + "nlp/v1/sentiment",
                      headers=headers,
                      data=json.dumps(data))
    return response.json()