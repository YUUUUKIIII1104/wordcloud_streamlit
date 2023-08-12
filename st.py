
import streamlit as st
from app import extract_text_from_pdf, create_wordcloud, extract_nouns, emotion_analysis

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

    # st.subheader("感情分析結果 (ML-Ask)")
    # emotion_results = emotion_analysis(extracted_text)
    # st.write(emotion_results)
