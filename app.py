import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Processor", page_icon="📊", layout="wide")

st.title("📊 Data Processor")
st.write("Veri dosyalarınızı yükleyin ve ayrıntılı analiz yapın")

uploaded_file = st.file_uploader("Dosya seçin", type=['csv', 'xlsx'])

if uploaded_file:
    st.success(f"Dosya yüklendi: {uploaded_file.name}")
    
    # Veri okuma
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Veri özeti
    st.subheader("📋 Veri Özeti")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Satır Sayısı", len(df))
    
    with col2:
        st.metric("Sütun Sayısı", len(df.columns))
    
    with col3:
        st.metric("Eksik Veri", df.isnull().sum().sum())
    
    with col4:
        st.metric("Bellek Kullanımı", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Eksik veri analizi
    st.subheader("🔍 Eksik Veri Analizi")
    
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        missing_df = pd.DataFrame({
            'Sütun': missing_data.index,
            'Eksik Veri': missing_data.values,
            'Yüzde (%)': (missing_data.values / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Eksik Veri'] > 0]
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ Eksik veri bulunmuyor!")
    
    # Sayısal sütunlar için temel istatistikler
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        st.subheader("📊 Sayısal Sütunlar İstatistikleri")
        st.dataframe(df[numeric_columns].describe(), use_container_width=True)
    
    # Veri önizleme
    st.subheader("👀 Veri Önizleme")
    
    # Görüntülenecek satır sayısı seçimi
    show_rows = st.slider("Görüntülenecek satır sayısı", 5, min(50, len(df)), 10)
    
    # Tam genişlik dataframe
    st.dataframe(df.head(show_rows), use_container_width=True, height=400)
    
    # Veri tipleri
    st.subheader("🏷️ Veri Tipleri")
    
    dtype_df = pd.DataFrame({
        'Sütun': df.dtypes.index,
        'Veri Tipi': df.dtypes.values
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        # Veri tipi dağılımı
        st.write("**Veri Tipi Dağılımı:**")
        type_counts = df.dtypes.value_counts()
        for dtype, count in type_counts.items():
            st.write(f"• {dtype}: {count} sütun")

else:
    st.info("👆 Lütfen analiz etmek istediğiniz CSV veya Excel dosyasını yükleyin")