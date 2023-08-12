
import re
from sudachipy import dictionary
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# from mlask import MLAsk

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
    wc = WordCloud(font_path='/Library/Fonts/Arial Unicode.ttf', stopwords=stopwords, background_color=background_color, max_words=max_words, width=width, height=height)
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


# def emotion_analysis(text):
#     """テキストをML-Askを使用して解析し、ラッセル円環モデルに基づいた感情分析を行う。

#     Args:
#     - text (str): 入力テキスト。
    
#     Returns:
#     - dict: 分析結果の辞書。
#     """
#     emotion_analyzer = MLAsk()
#     return emotion_analyzer.analyze(text)


import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def preprocess_text_japanese(text):
    # 1. テキストの正規化
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    # 2. トークン化
    sudachi_tokenizer = dictionary.Dictionary().create()
    tokens = [m.surface() for m in sudachi_tokenizer.tokenize(text, mode=Tokenizer.SplitMode.C)]

    # 3. ストップワードの削除
    stopwords = set()  # ここに日本語のストップワードを追加
    tokens = [token for token in tokens if token not in stopwords]

    # 4. ステミング/ルンマタイゼーション
    tokens = [m.dictionary_form() for m in sudachi_tokenizer.tokenize(text, mode=Tokenizer.SplitMode.C)]

    return ' '.join(tokens)

def analyze_sentiment_japanese(text):
    model_name = "bandainamco-mirai/distilbert-base-japanese"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model.cuda()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    if torch.cuda.is_available():
        inputs = {key: tensor.cuda() for key, tensor in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    _, preds = torch.max(logits, dim=1)
    label = "positive" if preds.item() == 1 else "negative"
    
    return label
