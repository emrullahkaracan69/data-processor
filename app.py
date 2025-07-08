import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Processor", page_icon="📊")

st.title("📊 Data Processor")
st.write("Veri dosyalarınızı yükleyin ve işleyin")

uploaded_file = st.file_uploader("Dosya seçin", type=['csv', 'xlsx'])

if uploaded_file:
    st.success(f"Dosya yüklendi: {uploaded_file.name}")
    
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.dataframe(df.head())
else:
    st.info("Lütfen bir dosya yükleyin")