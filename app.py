import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import io

st.set_page_config(page_title="Data Processor", page_icon="📊", layout="wide")

st.title("📊 Data Processor")
st.write("Kapsamlı veri analizi ve eksik değer çözümleri")

# Fonksiyonları tanımlayalım
def grabColNames(dataframe, catTh=10, carTh=20):
    """Veri setindeki kategorik, numerik ve kardinal değişkenleri ayırır"""
    catCols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    numButCat = [col for col in dataframe.columns if dataframe[col].nunique() < catTh and
                   dataframe[col].dtypes != "O"]
    catButCar = [col for col in dataframe.columns if dataframe[col].nunique() > carTh and
                   dataframe[col].dtypes == "O"]
    catCols = catCols + numButCat
    catCols = [col for col in catCols if col not in catButCar]
    
    numCols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    numCols = [col for col in numCols if col not in numButCat]
    
    return catCols, numCols, catButCar

def missingValuesTable(dataframe, naName=False):
    """Eksik değer analizi"""
    naColumns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    
    if len(naColumns) == 0:
        return pd.DataFrame(), []
    
    nMiss = dataframe[naColumns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[naColumns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missingDf = pd.concat([nMiss, np.round(ratio, 2)], axis=1, keys=['Eksik_Sayi', 'Yuzde'])
    
    if naName:
        return missingDf, naColumns
    return missingDf

def missingVsTarget(dataframe, target, naColumns):
    """Eksik değerlerin hedef değişken ile analizi"""
    tempDf = dataframe.copy()
    results = {}
    
    for col in naColumns:
        tempDf[col + '_NA_FLAG'] = np.where(tempDf[col].isnull(), 1, 0)
    
    naFlags = tempDf.loc[:, tempDf.columns.str.contains("_NA_")].columns
    
    for col in naFlags:
        result = pd.DataFrame({
            "TARGET_MEAN": tempDf.groupby(col)[target].mean(),
            "Count": tempDf.groupby(col)[target].count()
        })
        results[col] = result
    
    return results

def convert_df_to_csv(df):
    """DataFrame'i CSV formatına çevirir"""
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    """DataFrame'i Excel formatına çevirir"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Processed_Data')
    return output.getvalue()

# Ana uygulama
uploaded_file = st.file_uploader("Dosya seçin", type=['csv', 'xlsx'])

if uploaded_file:
    st.success(f"Dosya yüklendi: {uploaded_file.name}")
    
    # Veri okuma
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # DataFrame'i session state'e kaydet
    if 'original_df' not in st.session_state:
        st.session_state.original_df = df.copy()
        st.session_state.processed_df = df.copy()
    
    # İşlenmiş veriyi kullan
    df = st.session_state.processed_df
    
    # Temel veri inceleme
    st.header("📋 Temel Veri İnceleme")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Satır Sayısı", df.shape[0])
    with col2:
        st.metric("Sütun Sayısı", df.shape[1])
    with col3:
        st.metric("Toplam Eksik Veri", df.isnull().sum().sum())
    
    # Veri önizleme
    st.subheader("👀 Veri Önizleme")
    show_rows = st.slider("Görüntülenecek satır sayısı", 5, min(50, len(df)), 10)
    st.dataframe(df.head(show_rows), use_container_width=True)
    
    # İstatistiksel özellikler
    st.subheader("📊 İstatistiksel Özellikler")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    # Unique değerler analizi
    st.header("🔍 Unique Değerler Analizi")
    st.write("**Her sütun için unique değer sayıları:**")
    
    unique_counts = pd.DataFrame({
        'Sütun': df.columns,
        'Unique_Sayi': [df[col].nunique() for col in df.columns],
        'Veri_Tipi': [str(df[col].dtype) for col in df.columns]
    }).sort_values('Unique_Sayi', ascending=False)
    
    st.dataframe(unique_counts, use_container_width=True)
    
    # Değişken tipi belirleme
    st.header("🏷️ Değişken Tipi Belirleme")
    
    col1, col2 = st.columns(2)
    with col1:
        cat_threshold = st.number_input("Kategorik eşik değeri (unique sayısı)", 1, 50, 10)
    with col2:
        car_threshold = st.number_input("Kardinal eşik değeri (unique sayısı)", 10, 100, 20)
    
    if st.button("Değişken Tiplerini Belirle"):
        catCols, numCols, catButCar = grabColNames(df, cat_threshold, car_threshold)
        
        st.success(f"**Analiz Sonuçları:**")
        st.write(f"• **Kategorik değişkenler:** {len(catCols)} adet")
        st.write(f"• **Numerik değişkenler:** {len(numCols)} adet")
        st.write(f"• **Kardinal değişkenler:** {len(catButCar)} adet")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📝 Kategorik Değişkenler:**")
            for col in catCols:
                st.write(f"• {col}")
        
        with col2:
            st.write("**🔢 Numerik Değişkenler:**")
            for col in numCols:
                st.write(f"• {col}")
        
        with col3:
            st.write("**🗂️ Kardinal Değişkenler:**")
            for col in catButCar:
                st.write(f"• {col}")
        
        # Session state'e kaydet
        st.session_state.catCols = catCols
        st.session_state.numCols = numCols
        st.session_state.catButCar = catButCar
    
    # Eksik değer analizi
    st.header("❌ Eksik Değer Analizi")
    
    missing_df, na_columns = missingValuesTable(df, naName=True)
    
    if len(na_columns) > 0:
        st.warning(f"**{len(na_columns)} sütunda eksik değer bulundu:**")
        st.dataframe(missing_df, use_container_width=True)
        
        # Eksik değerlerin hedef değişken ile analizi
        st.subheader("🎯 Eksik Değerlerin Hedef Değişken ile Analizi")
        
        target_col = st.selectbox("Hedef değişkeni seçin:", df.columns)
        
        if target_col and st.button("Eksik Değer vs Hedef Analizi"):
            results = missingVsTarget(df, target_col, na_columns)
            
            for col_name, result_df in results.items():
                st.write(f"**{col_name}:**")
                st.dataframe(result_df)
        
        # Eksik değer çözüm yöntemleri
        st.header("🔧 Eksik Değer Çözüm Yöntemleri")
        
        solution_method = st.selectbox(
            "Çözüm yöntemini seçin:",
            [
                "Seçim yapın...",
                "Yöntem 1: Silmek",
                "Yöntem 2: Basit Atama",
                "Yöntem 3: Kategorik Kırılımında Atama",
                "Yöntem 4: Tahmine Dayalı Atama (KNN)"
            ]
        )
        
        if solution_method == "Yöntem 1: Silmek":
            st.info("**Açıklama:** Eksik değeri olan satırları tamamen siler.")
            st.warning(f"**Mevcut satır sayısı:** {len(df)}")
            st.warning(f"**Silindikten sonra kalan satır sayısı:** {len(df.dropna())}")
            
            if st.button("Eksik Değerli Satırları Sil"):
                st.session_state.processed_df = df.dropna()
                st.success(f"✅ {len(df) - len(st.session_state.processed_df)} satır silindi!")
                st.experimental_rerun()
        
        elif solution_method == "Yöntem 2: Basit Atama":
            st.info("**Açıklama:** Eksik değerleri basit istatistiksel yöntemlerle doldurur.")
            
            selected_column = st.selectbox("Doldurmak istediğiniz sütunu seçin:", na_columns)
            
            if selected_column:
                col_dtype = df[selected_column].dtype
                
                if col_dtype in ['int64', 'float64']:  # Numerik
                    fill_method = st.selectbox(
                        "Doldurma yöntemini seçin:",
                        ["Ortalama", "Medyan", "Sabit Değer"]
                    )
                    
                    if fill_method == "Ortalama":
                        fill_value = df[selected_column].mean()
                        st.info(f"Ortalama değer: {fill_value:.2f}")
                    elif fill_method == "Medyan":
                        fill_value = df[selected_column].median()
                        st.info(f"Medyan değer: {fill_value:.2f}")
                    elif fill_method == "Sabit Değer":
                        fill_value = st.number_input("Sabit değer girin:")
                    
                    if st.button("Uygula"):
                        st.session_state.processed_df[selected_column] = df[selected_column].fillna(fill_value)
                        st.success(f"✅ {selected_column} sütunu dolduruldu!")
                        st.experimental_rerun()
                
                else:  # Kategorik
                    fill_method = st.selectbox(
                        "Doldurma yöntemini seçin:",
                        ["Mod (En Sık Değer)", "Önceki Satır", "Sonraki Satır", "Sabit Değer"]
                    )
                    
                    if fill_method == "Mod (En Sık Değer)":
                        if len(df[selected_column].mode()) > 0:
                            fill_value = df[selected_column].mode()[0]
                            st.info(f"Mod değer: {fill_value}")
                        else:
                            st.error("Mod değer bulunamadı!")
                    elif fill_method == "Sabit Değer":
                        fill_value = st.text_input("Sabit değer girin:")
                    
                    if st.button("Uygula"):
                        if fill_method == "Önceki Satır":
                            st.session_state.processed_df[selected_column] = df[selected_column].fillna(method='ffill')
                        elif fill_method == "Sonraki Satır":
                            st.session_state.processed_df[selected_column] = df[selected_column].fillna(method='bfill')
                        else:
                            st.session_state.processed_df[selected_column] = df[selected_column].fillna(fill_value)
                        
                        st.success(f"✅ {selected_column} sütunu dolduruldu!")
                        st.experimental_rerun()
        
        elif solution_method == "Yöntem 3: Kategorik Kırılımında Atama":
            st.info("**Açıklama:** Eksik değerleri başka bir kategorik değişkene göre gruplandırarak doldurur.")
            
            if 'catCols' in st.session_state:
                numeric_na_cols = [col for col in na_columns if col in st.session_state.numCols]
                
                if len(numeric_na_cols) > 0:
                    selected_target = st.selectbox("Doldurulacak numerik sütun:", numeric_na_cols)
                    selected_group = st.selectbox("Gruplandırma sütunu:", st.session_state.catCols)
                    
                    if selected_target and selected_group:
                        group_means = df.groupby(selected_group)[selected_target].mean()
                        st.write("**Grup ortalama değerleri:**")
                        st.dataframe(group_means.to_frame("Ortalama"))
                        
                        if st.button("Gruplandırılmış Ortalama ile Doldur"):
                            st.session_state.processed_df[selected_target] = df[selected_target].fillna(df.groupby(selected_group)[selected_target].transform('mean'))
                            st.success(f"✅ {selected_target} sütunu {selected_group} gruplarının ortalamasıyla dolduruldu!")
                            st.experimental_rerun()
                else:
                    st.warning("Numerik eksik değerli sütun bulunamadı!")
            else:
                st.warning("Önce değişken tiplerini belirleyin!")
        
        elif solution_method == "Yöntem 4: Tahmine Dayalı Atama (KNN)":
            st.info("**Açıklama:** K-Nearest Neighbors algoritmasıyla eksik değerleri tahmin eder.")
            
            if 'catCols' in st.session_state:
                k_neighbors = st.slider("K (komşu sayısı)", 3, 10, 5)
                
                if st.button("KNN ile Doldur"):
                    try:
                        # Encoding ve scaling
                        catCols = st.session_state.catCols
                        numCols = st.session_state.numCols
                        
                        # Get dummies
                        df_encoded = pd.get_dummies(df[catCols + numCols], drop_first=True)
                        
                        # Scaling
                        scaler = MinMaxScaler()
                        df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
                        
                        # KNN Imputation
                        imputer = KNNImputer(n_neighbors=k_neighbors)

                        df_imputed = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df_scaled.columns)

                        

                        # Eski haline çevirme

                        df_final = pd.DataFrame(scaler.inverse_transform(df_imputed), columns=df_scaled.columns)

                        

                        # Orijinal DataFrame'e geri koyma

                        for col in numCols:

                            if col in df_final.columns:

                                st.session_state.processed_df[col] = df_final[col]

                        

                        st.success("✅ KNN algoritmasıyla eksik değerler dolduruldu!")

                        st.experimental_rerun()

                        

                    except Exception as e:

                        st.error(f"Hata: {str(e)}")

                        st.warning("KNN yöntemi için tüm kategorik değişkenler sayısal değerlere dönüştürülmelidir.")

            else:

                st.warning("Önce değişken tiplerini belirleyin!")

    

    else:

        st.success("✅ Eksik değer bulunmuyor!")

    

    # Veri İndirme Bölümü

    st.header("💾 İşlenmiş Veriyi İndir")

    

    col1, col2, col3 = st.columns(3)

    

    with col1:

        # Orijinal veriyi sıfırla

        if st.button("🔄 Orijinal Veriye Sıfırla"):

            st.session_state.processed_df = st.session_state.original_df.copy()

            st.success("✅ Veriler orijinal haline sıfırlandı!")

            st.experimental_rerun()

    

    with col2:

        # CSV olarak indir

        csv_data = convert_df_to_csv(st.session_state.processed_df)

        st.download_button(

            label="📥 CSV olarak İndir",

            data=csv_data,

            file_name=f"processed_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",

            mime="text/csv"

        )

    

    with col3:

        # Excel olarak indir

        excel_data = convert_df_to_excel(st.session_state.processed_df)

        st.download_button(

            label="📥 Excel olarak İndir",

            data=excel_data,

            file_name=f"processed_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",

            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        )

    

    # Değişikliklerin özeti

    st.header("📋 Değişikliklerin Özeti")

    

    col1, col2 = st.columns(2)

    

    with col1:

        st.subheader("🔵 Orijinal Veri")

        st.write(f"**Satır sayısı:** {st.session_state.original_df.shape[0]}")

        st.write(f"**Sütun sayısı:** {st.session_state.original_df.shape[1]}")

        st.write(f"**Eksik değer:** {st.session_state.original_df.isnull().sum().sum()}")

    

    with col2:

        st.subheader("🟢 İşlenmiş Veri")

        st.write(f"**Satır sayısı:** {st.session_state.processed_df.shape[0]}")

        st.write(f"**Sütun sayısı:** {st.session_state.processed_df.shape[1]}")

        st.write(f"**Eksik değer:** {st.session_state.processed_df.isnull().sum().sum()}")

    

    # Değişikliklerin detayı

    if st.session_state.original_df.shape[0] != st.session_state.processed_df.shape[0]:

        st.info(f"ℹ️ {st.session_state.original_df.shape[0] - st.session_state.processed_df.shape[0]} satır silindi.")

    

    if st.session_state.original_df.isnull().sum().sum() != st.session_state.processed_df.isnull().sum().sum():

        st.info(f"ℹ️ {st.session_state.original_df.isnull().sum().sum() - st.session_state.processed_df.isnull().sum().sum()} eksik değer işlendi.")


else:

    st.info("👆 Lütfen analiz etmek istediğiniz CSV veya Excel dosyasını yükleyin")

    

    # Örnek veri seti önerileri

    st.header("📊 Örnek Veri Setleri")

    

    col1, col2, col3 = st.columns(3)

    

    with col1:

        st.info("**Titanic Veri Seti**\n\nEksik değerler içeren klasik veri seti. Age, Cabin, Embarked sütunlarında eksik değerler var.")

    

    with col2:

        st.info("**House Prices Veri Seti**\n\nGayrimenkul fiyatları verisi. Çok sayıda kategorik ve numerik değişken içerir.")

    

    with col3:

        st.info("**Customer Data**\n\nMüşteri verisi. Demografik bilgiler ve satın alma davranışları.")