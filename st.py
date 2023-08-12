import streamlit as st
import pandas as pd
from app import extract_text_from_pdf, create_wordcloud, extract_nouns, get_cotoha_acces_token, cotoha_sentiment_analyze

st.title("PDFテキストマイニングアプリ")

uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type=["pdf"])

if uploaded_file:
    with st.spinner('PDFからテキストを抽出中...'):
        extracted_text = extract_text_from_pdf(uploaded_file)

    # st.subheader("抽出されたテキスト")
    # st.write(extracted_text[:1000])  # 最初の1000文字だけ表示
    
    # st.subheader("全テキストのワードクラウド")
    # fig = create_wordcloud(extracted_text)
    # st.pyplot(fig)

    # st.subheader("名詞の抽出")
    nouns = extract_nouns(extracted_text)
    # st.write(', '.join(nouns[:50]))  # 最初の50個の名詞だけ表示

    st.subheader("名詞のワードクラウド")
    fig_nouns = create_wordcloud(' '.join(nouns))
    st.pyplot(fig_nouns)
    
    st.subheader("感情分析結果")
    accses_token = get_cotoha_acces_token()
    response = cotoha_sentiment_analyze(accses_token, extracted_text)

    sentiment = response["result"]["sentiment"]
    score = response["result"]["score"]
    emotional_phrase = response["result"]["emotional_phrase"]
    df_emotional_phrase = pd.DataFrame(emotional_phrase)
    st.write(f'## 分析結果:{sentiment}')
    #st.write(f'### スコア:{score}')
    st.dataframe(df_emotional_phrase)


