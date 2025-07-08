import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Data Processor", page_icon="📊", layout="wide")

st.title("📊 Data Processor")
st.write("Veri dosyalarınızı yükleyin ve ayrıntılı analiz yapın")

# Fonksiyonları tanımlayalım
def checkDf(dataframe, head=5):
    """Temel veri inceleme"""
    results = {}
    results['shape'] = dataframe.shape
    results['dtypes'] = dataframe.dtypes
    results['head'] = dataframe.head(head)
    results['tail'] = dataframe.tail(head)
    results['null_sum'] = dataframe.isnull().sum()
    results['describe'] = dataframe.describe([0, 0.05, 0.50, 0.95, 1]).T
    return results

def grabColNames(dataframe, catTH=10, carTH=20):
    """Değişken tiplerini ayırma"""
    catCols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    numButCat = [col for col in dataframe.columns if dataframe[col].nunique() < catTH and dataframe[col].dtypes in ["int64", "float64"]]
    catButCar = [col for col in dataframe.columns if dataframe[col].nunique() > carTH and str(dataframe[col].dtypes) in ["category", "object"]]
    
    catCols = catCols + numButCat
    catCols = [col for col in catCols if col not in catButCar]
    
    numCols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    numCols = [col for col in numCols if col not in catCols]
    
    return catCols, numCols, catButCar

def catSummary(dataframe, colName):
    """Kategorik değişken özeti"""
    summary = pd.DataFrame({
        colName: dataframe[colName].value_counts(),
        "Ratio": 100 * dataframe[colName].value_counts() / len(dataframe)
    })
    return summary

def numSummary(dataframe, numericalCol):
    """Numerik değişken özeti"""
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    return dataframe[numericalCol].describe(quantiles)

def targetSummaryWithCat(dataframe, target, categoricalCol):
    """Hedef değişken ile kategorik analiz"""
    return pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categoricalCol)[target].mean()})

def targetSummaryWithNum(dataframe, target, numericalCol):
    """Hedef değişken ile numerik analiz"""
    return dataframe.groupby(target).agg({numericalCol: "mean"})

def highCorrelatedCols(dataframe, corrTh=0.90):
    """Yüksek korelasyonlu sütunları bulma"""
    numCols = [col for col in dataframe.columns if dataframe[col].dtype in ['int64', 'float64']]
    if len(numCols) < 2:
        return []
    
    corr = dataframe[numCols].corr()
    corMatrix = corr.abs()
    upperTriangleMatrix = corMatrix.where(np.triu(np.ones(corMatrix.shape), k=1).astype(bool))
    dropList = [col for col in upperTriangleMatrix.columns if any(upperTriangleMatrix[col] > corrTh)]
    return dropList, corr

# Ana uygulama
uploaded_file = st.file_uploader("Dosya seçin", type=['csv', 'xlsx'])

if uploaded_file:
    st.success(f"Dosya yüklendi: {uploaded_file.name}")
    
    # Veri okuma
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Temel veri inceleme
    st.header("🔍 Temel Veri İnceleme")
    
    if st.checkbox("Veri setini incele (checkDf)"):
        results = checkDf(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Satır Sayısı", results['shape'][0])
        with col2:
            st.metric("Sütun Sayısı", results['shape'][1])
        
        st.subheader("📋 Veri Tipleri")
        st.dataframe(pd.DataFrame(results['dtypes']).rename(columns={0: 'Veri Tipi'}))
        
        st.subheader("👀 İlk 5 Satır")
        st.dataframe(results['head'])
        
        st.subheader("👀 Son 5 Satır")
        st.dataframe(results['tail'])
        
        st.subheader("❌ Eksik Değerler")
        missing_df = pd.DataFrame(results['null_sum']).rename(columns={0: 'Eksik Değer'})
        missing_df = missing_df[missing_df['Eksik Değer'] > 0]
        if len(missing_df) > 0:
            st.dataframe(missing_df)
        else:
            st.success("Eksik değer bulunmuyor!")
        
        st.subheader("📊 İstatistiksel Özet")
        st.dataframe(results['describe'])
    
    # Değişken tiplerini ayırma
    st.header("🏷️ Değişken Tipi Analizi")
    
    if st.checkbox("Değişken tiplerini ayır (grabColNames)"):
        catCols, numCols, catButCar = grabColNames(df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📝 Kategorik Değişkenler")
            st.write(f"**Toplam: {len(catCols)}**")
            for col in catCols:
                st.write(f"• {col}")
        
        with col2:
            st.subheader("🔢 Numerik Değişkenler")
            st.write(f"**Toplam: {len(numCols)}**")
            for col in numCols:
                st.write(f"• {col}")
        
        with col3:
            st.subheader("🗂️ Kardinal Değişkenler")
            st.write(f"**Toplam: {len(catButCar)}**")
            for col in catButCar:
                st.write(f"• {col}")
        
        # Kategorik değişken analizi
        if len(catCols) > 0:
            st.subheader("📊 Kategorik Değişken Analizi")
            selected_cat = st.selectbox("Analiz etmek istediğiniz kategorik değişkeni seçin:", catCols)
            if selected_cat:
                cat_result = catSummary(df, selected_cat)
                st.dataframe(cat_result)
        
        # Numerik değişken analizi
        if len(numCols) > 0:
            st.subheader("🔢 Numerik Değişken Analizi")
            selected_num = st.selectbox("Analiz etmek istediğiniz numerik değişkeni seçin:", numCols)
            if selected_num:
                num_result = numSummary(df, selected_num)
                st.dataframe(pd.DataFrame(num_result).rename(columns={selected_num: 'Değer'}))
    
    # Hedef değişken analizi
    st.header("🎯 Hedef Değişken Analizi")
    
    if st.checkbox("Hedef değişken analizi yapmak istiyor musunuz?"):
        target_col = st.selectbox("Hedef değişkeni seçin:", df.columns)
        
        if target_col:
            catCols, numCols, catButCar = grabColNames(df)
            
            # Kategorik ile hedef analiz
            if len(catCols) > 0:
                st.subheader("📊 Kategorik Değişkenler ile Hedef Analizi")
                for col in catCols:
                    if col != target_col:
                        with st.expander(f"📋 {col} - {target_col} Analizi"):
                            target_cat_result = targetSummaryWithCat(df, target_col, col)
                            st.dataframe(target_cat_result)
            
            # Numerik ile hedef analiz
            if len(numCols) > 0:
                st.subheader("🔢 Numerik Değişkenler ile Hedef Analizi")
                for col in numCols:
                    if col != target_col:
                        with st.expander(f"📊 {col} - {target_col} Analizi"):
                            target_num_result = targetSummaryWithNum(df, target_col, col)
                            st.dataframe(target_num_result)
    
    # Korelasyon analizi
    st.header("🔗 Korelasyon Analizi")
    
    if st.checkbox("Yüksek korelasyonlu değişkenleri bulmak istiyor musunuz?"):
        corr_threshold = st.slider("Korelasyon eşiği", 0.5, 0.99, 0.90, 0.05)
        
        drop_list, corr_matrix = highCorrelatedCols(df, corr_threshold)
        
        if len(drop_list) > 0:
            st.warning(f"**Yüksek korelasyonlu değişkenler bulundu (>{corr_threshold}):**")
            for col in drop_list:
                st.write(f"• {col}")
            
            if st.checkbox("Korelasyon matrisini göster"):
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap="RdBu", center=0, ax=ax)
                st.pyplot(fig)
            
            if st.button("Yüksek korelasyonlu değişkenleri sil"):
                df_new = df.drop(drop_list, axis=1)
                st.success(f"{len(drop_list)} değişken silindi!")
                st.dataframe(df_new.head())
        else:
            st.success("Yüksek korelasyonlu değişken bulunamadı!")
    
    # Ham veri görüntüleme
    st.header("📋 Ham Veri")
    show_rows = st.slider("Görüntülenecek satır sayısı", 5, min(100, len(df)), 10)
    st.dataframe(df.head(show_rows), use_container_width=True)

else:
    st.info("👆 Lütfen analiz etmek istediğiniz CSV veya Excel dosyasını yükleyin")