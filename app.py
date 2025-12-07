# app.py - Customer Segmentation dengan PySpark MLlib
# ====================================================

# IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
if os.environ.get('JAVA_HOME') is None:
    # Path default untuk Java 17 di Debian/Streamlit Cloud
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-17-openjdk-amd64'
from datetime import datetime, timedelta

# PySpark Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum, countDistinct, max, min, avg, datediff,
    lit, to_timestamp, mean, when, count, round
)
from pyspark.sql.types import DoubleType, IntegerType, TimestampType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline

# ====================================================
# KONFIGURASI HALAMAN STREAMLIT
# ====================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================
# CSS CUSTOM DAN STYLING
# ====================================================

st.markdown("""
<style>
    /* MAIN TITLE */
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }

    /* SECTION TITLES */
    .section-title {
        font-size: 1.8rem;
        color: #2563EB;
        border-left: 5px solid #3B82F6;
        padding-left: 1rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    /* METRIC CARDS */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #E5E7EB;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }

    /* CLUSTER BOXES */
    .cluster-box-vip {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #000;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #FFD700;
    }
    .cluster-box-loyal {
        background: linear-gradient(135deg, #C0C0C0 0%, #808080 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #C0C0C0;
    }
    .cluster-box-regular {
        background: linear-gradient(135deg, #CD7F32 0%, #8B4513 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #CD7F32;
    }
    .cluster-box-inactive {
        background: linear-gradient(135deg, #A9A9A9 0%, #696969 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #A9A9A9;
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(90deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
    }

    /* PROGRESS BAR */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3B82F6 0%, #10B981 100%);
    }

    /* EXPANDER */
    .streamlit-expanderHeader {
        background-color: #F3F4F6;
        border-radius: 5px;
        font-weight: 600;
    }

    /* TABLES */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }

    /* SIDEBAR */
    .css-1d391kg {
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)


# ====================================================
# FUNGSI UTAMA - DATA PROCESSING
# ====================================================

@st.cache_resource
def init_spark_session():
    """Initialize Spark Session dengan konfigurasi optimal"""
    try:
        spark = SparkSession.builder \
            .appName("CustomerSegmentationApp") \
            .config("spark.driver.memory", "1g") \
            .config("spark.executor.memory", "1g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "4") \
            .getOrCreate()
            # Catatan: shuffle.partitions dikurangi jadi 4 biar lebih ringan

        spark.sparkContext.setLogLevel("WARN")
        return spark
    except Exception as e:
        st.error(f"Gagal menginisialisasi Spark: {str(e)}")
        st.info("Pastikan Java 8 atau 11 sudah terinstall di sistem Anda.")
        return None


def load_data(file_path, spark):
    """Load data dari file CSV"""
    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def preprocess_data(df, spark):
    """Preprocessing data: cleaning, transformasi, dll."""

    # Simpan original count
    original_count = df.count()

    # 1. Hapus CustomerID null
    df_clean = df.filter(col("CustomerID").isNotNull())

    # 2. Hapus Quantity <= 0 (retur/negatif)
    df_clean = df_clean.filter(col("Quantity") > 0)

    # 3. Hapus UnitPrice <= 0
    df_clean = df_clean.filter(col("UnitPrice") > 0)

    # 4. Konversi InvoiceDate ke timestamp
    # Coba format US (Bulan/Tanggal) dulu, kalau gagal baru format Indo (Tanggal/Bulan)
    df_clean = df_clean.withColumn(
        "InvoiceDate",
        coalesce(
            to_timestamp(col("InvoiceDate"), "M/d/yyyy H:mm"),
            to_timestamp(col("InvoiceDate"), "d/M/yyyy H:mm"),
            to_timestamp(col("InvoiceDate"), "yyyy-MM-dd HH:mm:ss")
        )
    )
    # 5. Hitung TotalAmount
    df_clean = df_clean.withColumn(
        "TotalAmount",
        round(col("Quantity") * col("UnitPrice"), 2)
    )

    # Hitung statistik cleaning
    cleaned_count = df_clean.count()
    removed_count = original_count - cleaned_count
    retention_rate = (cleaned_count / original_count) * 100 if original_count > 0 else 0

    return df_clean, {
        'original_count': original_count,
        'cleaned_count': cleaned_count,
        'removed_count': removed_count,
        'retention_rate': retention_rate
    }


def calculate_rfm(df_clean):
    """Hitung RFM metrics untuk setiap customer"""

    # Cari tanggal terakhir sebagai snapshot date
    max_date = df_clean.agg(max("InvoiceDate")).collect()[0][0]

    # Hitung RFM
    rfm_df = df_clean.groupBy("CustomerID").agg(
        datediff(lit(max_date), max("InvoiceDate")).alias("Recency"),
        countDistinct("InvoiceNo").alias("Frequency"),
        sum("TotalAmount").alias("Monetary"),
        count("InvoiceNo").alias("TotalTransactions"),
        avg("TotalAmount").alias("AvgTransactionValue"),
        min("InvoiceDate").alias("FirstPurchaseDate"),
        max("InvoiceDate").alias("LastPurchaseDate")
    )

    # Tambahkan segmentasi sederhana berdasarkan quartile
    from pyspark.sql.window import Window
    from pyspark.sql.functions import ntile

    window = Window.orderBy("Recency")
    rfm_df = rfm_df.withColumn("R_Quartile",
                               ntile(4).over(window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))

    window = Window.orderBy("Frequency")
    rfm_df = rfm_df.withColumn("F_Quartile",
                               ntile(4).over(window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))

    window = Window.orderBy("Monetary")
    rfm_df = rfm_df.withColumn("M_Quartile",
                               ntile(4).over(window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))

    # RFM Score
    rfm_df = rfm_df.withColumn("RFM_Score",
                               col("R_Quartile") * 100 + col("F_Quartile") * 10 + col("M_Quartile"))

    return rfm_df, max_date


def calculate_elbow_method(final_data, max_k=10):
    """Hitung WSSSE untuk Elbow Method"""
    errors = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(featuresCol="features", k=k, seed=42, maxIter=20)
        model = kmeans.fit(final_data)
        wssse = model.summary.trainingCost
        errors.append((k, wssse))

    return errors


def perform_clustering(rfm_df, k=4):
    """Lakukan clustering dengan K-Means dan GMM"""

    # 1. Vector Assembly
    assembler = VectorAssembler(
        inputCols=["Recency", "Frequency", "Monetary"],
        outputCol="features_vec"
    )

    # 2. Standard Scaling
    scaler = StandardScaler(
        inputCol="features_vec",
        outputCol="features",
        withStd=True,
        withMean=True
    )

    # 3. Pipeline
    pipeline = Pipeline(stages=[assembler, scaler])
    pipeline_model = pipeline.fit(rfm_df)
    final_data = pipeline_model.transform(rfm_df)

    # 4. K-Means Clustering
    kmeans = KMeans(
        featuresCol="features",
        k=k,
        seed=42,
        maxIter=30,
        tol=1e-4
    )
    kmeans_model = kmeans.fit(final_data)
    predictions_kmeans = kmeans_model.transform(final_data)

    # 5. Gaussian Mixture Model
    gmm = GaussianMixture(
        featuresCol="features",
        k=k,
        seed=42,
        maxIter=50,
        tol=1e-4
    )
    gmm_model = gmm.fit(final_data)
    predictions_gmm = gmm_model.transform(final_data)

    # 6. Evaluate Models
    evaluator = ClusteringEvaluator(
        featuresCol="features",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )

    score_kmeans = evaluator.evaluate(predictions_kmeans)
    score_gmm = evaluator.evaluate(predictions_gmm)

    # 7. Pilih model terbaik
    if score_kmeans > score_gmm:
        best_model = "K-Means"
        best_predictions = predictions_kmeans
        best_score = score_kmeans
        model_object = kmeans_model
    else:
        best_model = "Gaussian Mixture Model (GMM)"
        best_predictions = predictions_gmm
        best_score = score_gmm
        model_object = gmm_model

    return {
        'kmeans': predictions_kmeans,
        'gmm': predictions_gmm,
        'best_model': best_model,
        'best_predictions': best_predictions,
        'best_score': best_score,
        'scores': {'kmeans': score_kmeans, 'gmm': score_gmm},
        'final_data': final_data,
        'model_object': model_object,
        'pipeline_model': pipeline_model
    }


def analyze_clusters(predictions_df):
    """Analisis dan interpretasi hasil clustering"""

    # Konversi ke pandas untuk analisis
    predictions_pd = predictions_df.select(
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "prediction",
        "TotalTransactions",
        "AvgTransactionValue",
        "RFM_Score"
    ).toPandas()

    # Hitung statistik per cluster
    cluster_stats = predictions_pd.groupby("prediction").agg({
        "CustomerID": "count",
        "Recency": ["mean", "std", "min", "max"],
        "Frequency": ["mean", "std", "min", "max"],
        "Monetary": ["mean", "std", "min", "max"],
        "TotalTransactions": "mean",
        "AvgTransactionValue": "mean",
        "RFM_Score": "mean"
    }).round(2)

    # Flatten column names
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
    cluster_stats = cluster_stats.reset_index()

    # Beri nama cluster berdasarkan karakteristik
    cluster_names = []
    cluster_descriptions = []
    cluster_colors = []

    for idx, row in cluster_stats.iterrows():
        recency = row['Recency_mean']
        frequency = row['Frequency_mean']
        monetary = row['Monetary_mean']
        count = row['CustomerID_count']

        # Tentukan segmentasi
        if recency < 50 and frequency > 10 and monetary > 2000:
            name = "💎 VIP Customers"
            desc = "Pelanggan sangat aktif dengan nilai tinggi"
            color = "cluster-box-vip"
        elif recency < 100 and frequency > 5 and monetary > 1000:
            name = "⭐ Loyal Customers"
            desc = "Pelanggan setia dengan frekuensi tinggi"
            color = "cluster-box-loyal"
        elif recency < 30:
            name = "🆕 New Customers"
            desc = "Pelanggan baru dengan potensi berkembang"
            color = "cluster-box-regular"
        elif recency > 180:
            name = "😴 Inactive Customers"
            desc = "Pelanggan tidak aktif > 6 bulan"
            color = "cluster-box-inactive"
        elif monetary > 500:
            name = "💰 High Value"
            desc = "Pelanggan dengan nilai transaksi tinggi"
            color = "cluster-box-vip"
        else:
            name = f"📊 Cluster {int(idx)}"
            desc = "Segment pelanggan reguler"
            color = "cluster-box-regular"

        cluster_names.append(name)
        cluster_descriptions.append(desc)
        cluster_colors.append(color)

    cluster_stats['Segment_Name'] = cluster_names
    cluster_stats['Description'] = cluster_descriptions
    cluster_stats['Color_Class'] = cluster_colors

    return predictions_pd, cluster_stats


def generate_recommendations(cluster_stats):
    """Generate rekomendasi bisnis berdasarkan cluster"""

    recommendations = {}

    for _, row in cluster_stats.iterrows():
        segment = row['Segment_Name']
        recency = row['Recency_mean']
        frequency = row['Frequency_mean']
        monetary = row['Monetary_mean']

        if "VIP" in segment:
            recs = [
                "🎁 Program loyalitas eksklusif dengan reward khusus",
                "🌟 Akses early-bird untuk produk baru dan pre-order",
                "👑 Dedicated customer service dengan priority support",
                "💎 Undangan ke event eksklusif dan preview produk",
                "📈 Personalisasi maksimal dengan rekomendasi premium"
            ]
        elif "Loyal" in segment:
            recs = [
                "💝 Program referral dengan komisi menarik",
                "🔥 Bundle produk dengan diskon khusus member",
                "📧 Personalized email marketing berdasarkan riwayat",
                "🎯 Upsell produk komplementer dengan package deal",
                "🏆 Tiered loyalty program untuk naik level"
            ]
        elif "New" in segment:
            recs = [
                "👋 Welcome series email dengan panduan produk",
                "🎯 Diskon 20% untuk transaksi kedua dalam 30 hari",
                "📚 Konten edukasi dan tips penggunaan produk",
                "🤝 Onboarding call dengan customer success team",
                "🔔 Reminder untuk first purchase anniversary"
            ]
        elif "Inactive" in segment:
            recs = [
                "📲 Re-engagement campaign dengan win-back offer",
                "💸 Special discount 30% untuk kembali berbelanja",
                "❓ Survey kepuasan dan feedback collection",
                "🎁 We miss you campaign dengan small gift",
                "📞 Outbound call dari customer service"
            ]
        else:
            recs = [
                "📊 Regular newsletter dengan promo bulanan",
                "🎯 Cross-selling berdasarkan purchase history",
                "📱 Engagement melalui social media dan content",
                "💡 Educational content tentang produk",
                "🔄 Reactivation campaign setiap 3 bulan"
            ]

        recommendations[segment] = recs

    return recommendations


# ====================================================
# FUNGSI VISUALISASI
# ====================================================

def plot_rfm_distributions(rfm_pd):
    """Plot distribusi RFM metrics"""

    fig = plt.figure(figsize=(15, 10))

    # Recency Distribution
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(rfm_pd['Recency'], bins=50, color='#3B82F6', edgecolor='black', alpha=0.7)
    ax1.set_title('Distribusi Recency', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Hari sejak transaksi terakhir')
    ax1.set_ylabel('Jumlah Pelanggan')
    ax1.grid(True, alpha=0.3)

    # Frequency Distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(rfm_pd[rfm_pd['Frequency'] < 100]['Frequency'], bins=50, color='#10B981', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribusi Frequency', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Jumlah transaksi')
    ax2.set_ylabel('Jumlah Pelanggan')
    ax2.grid(True, alpha=0.3)

    # Monetary Distribution
    ax3 = plt.subplot(2, 3, 3)
    monetary_filtered = rfm_pd[rfm_pd['Monetary'] < rfm_pd['Monetary'].quantile(0.95)]
    ax3.hist(monetary_filtered['Monetary'], bins=50, color='#8B5CF6', edgecolor='black', alpha=0.7)
    ax3.set_title('Distribusi Monetary', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Total pengeluaran ($)')
    ax3.set_ylabel('Jumlah Pelanggan')
    ax3.grid(True, alpha=0.3)

    # RFM Scatter: Recency vs Frequency
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(rfm_pd['Recency'], rfm_pd['Frequency'],
                          c=rfm_pd['Monetary'], cmap='viridis',
                          alpha=0.6, s=20)
    ax4.set_title('Recency vs Frequency', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Recency (hari)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Monetary ($)')

    # RFM Scatter: Frequency vs Monetary
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(rfm_pd['Frequency'], rfm_pd['Monetary'],
                          c=rfm_pd['Recency'], cmap='plasma',
                          alpha=0.6, s=20)
    ax5.set_title('Frequency vs Monetary', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Frequency')
    ax5.set_ylabel('Monetary ($)')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Recency (hari)')

    # RFM Scatter: Recency vs Monetary
    ax6 = plt.subplot(2, 3, 6)
    scatter = ax6.scatter(rfm_pd['Recency'], rfm_pd['Monetary'],
                          c=rfm_pd['Frequency'], cmap='coolwarm',
                          alpha=0.6, s=20)
    ax6.set_title('Recency vs Monetary', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Recency (hari)')
    ax6.set_ylabel('Monetary ($)')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Frequency')

    plt.tight_layout()
    return fig


def plot_elbow_curve(errors):
    """Plot elbow curve untuk menentukan optimal k"""

    k_values = [e[0] for e in errors]
    wssse_values = [e[1] for e in errors]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, wssse_values, 'bo-', linewidth=2, markersize=8, markerfacecolor='red')

    # Highlight optimal k (elbow point)
    diffs = np.diff(wssse_values)
    diff_ratios = diffs[:-1] / diffs[1:]
    if len(diff_ratios) > 0:
        optimal_idx = np.argmax(diff_ratios) + 1
        optimal_k = k_values[optimal_idx]
        ax.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'Optimal k ≈ {optimal_k}')

    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Within Sum of Squared Errors (WSSSE)', fontsize=12)
    ax.set_title('Elbow Method for Optimal k Selection', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate each point
    for k, wssse in zip(k_values, wssse_values):
        ax.annotate(f'{wssse:.0f}', (k, wssse), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    return fig


def plot_cluster_3d(predictions_pd):
    """Plot 3D visualization clusters"""

    # Create 3D scatter plot
    fig = px.scatter_3d(
        predictions_pd,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='prediction',
        title='🎯 3D Visualization of Customer Segments',
        labels={
            'Recency': 'Recency (Days)',
            'Frequency': 'Frequency',
            'Monetary': 'Monetary ($)',
            'prediction': 'Cluster'
        },
        opacity=0.7,
        color_continuous_scale=px.colors.sequential.Viridis,
        hover_data=['CustomerID', 'TotalTransactions', 'AvgTransactionValue'],
        size='Monetary',
        size_max=20
    )

    fig.update_traces(
        marker=dict(
            size=5,
            line=dict(width=0.5, color='DarkSlateGrey')
        ),
        selector=dict(mode='markers')
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='Recency (Hari sejak pembelian terakhir)',
            yaxis_title='Frequency (Jumlah transaksi)',
            zaxis_title='Monetary (Total pengeluaran $)'
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    return fig


def plot_cluster_comparison(predictions_pd):
    """Plot perbandingan antar cluster"""

    fig = plt.figure(figsize=(15, 10))

    # Cluster Size
    ax1 = plt.subplot(2, 3, 1)
    cluster_counts = predictions_pd['prediction'].value_counts().sort_index()
    colors = plt.cm.Set3(np.arange(len(cluster_counts)) / len(cluster_counts))
    ax1.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors, edgecolor='black')
    ax1.set_title('Jumlah Pelanggan per Cluster', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Jumlah Pelanggan')
    ax1.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(cluster_counts.values):
        ax1.text(i, v + max(cluster_counts.values) * 0.01, str(v), ha='center', fontweight='bold')

    # Average Recency per Cluster
    ax2 = plt.subplot(2, 3, 2)
    recency_means = predictions_pd.groupby('prediction')['Recency'].mean()
    ax2.bar(recency_means.index.astype(str), recency_means.values,
            color='#FF6B6B', edgecolor='black')
    ax2.set_title('Rata-rata Recency per Cluster', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Rata-rata Recency (hari)')
    ax2.grid(True, alpha=0.3, axis='y')

    # Average Frequency per Cluster
    ax3 = plt.subplot(2, 3, 3)
    frequency_means = predictions_pd.groupby('prediction')['Frequency'].mean()
    ax3.bar(frequency_means.index.astype(str), frequency_means.values,
            color='#4ECDC4', edgecolor='black')
    ax3.set_title('Rata-rata Frequency per Cluster', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Rata-rata Frequency')
    ax3.grid(True, alpha=0.3, axis='y')

    # Average Monetary per Cluster
    ax4 = plt.subplot(2, 3, 4)
    monetary_means = predictions_pd.groupby('prediction')['Monetary'].mean()
    ax4.bar(monetary_means.index.astype(str), monetary_means.values,
            color='#FFD166', edgecolor='black')
    ax4.set_title('Rata-rata Monetary per Cluster', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Rata-rata Monetary ($)')
    ax4.grid(True, alpha=0.3, axis='y')

    # Box Plot Recency
    ax5 = plt.subplot(2, 3, 5)
    predictions_pd.boxplot(column='Recency', by='prediction', ax=ax5,
                           grid=True, patch_artist=True,
                           boxprops=dict(facecolor='#118AB2', alpha=0.7))
    ax5.set_title('Distribusi Recency per Cluster', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Cluster')
    ax5.set_ylabel('Recency (hari)')

    # Box Plot Monetary
    ax6 = plt.subplot(2, 3, 6)
    predictions_pd.boxplot(column='Monetary', by='prediction', ax=ax6,
                           grid=True, patch_artist=True,
                           boxprops=dict(facecolor='#06D6A0', alpha=0.7))
    ax6.set_title('Distribusi Monetary per Cluster', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Cluster')
    ax6.set_ylabel('Monetary ($)')

    plt.suptitle('Perbandingan Karakteristik Antar Cluster', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


# ====================================================
# MAIN APPLICATION
# ====================================================

def main():
    # HEADER
    st.markdown('<h1 class="main-title">🎯 SISTEM SEGMENTASI PELANGGAN RFM</h1>', unsafe_allow_html=True)
    st.markdown("### PENERAPAN MLIB PADA PYSPARK UNTUK SEGMENTASI PELANGGAN BERBASIS DATA TRANSAKSI")

    # SIDEBAR
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/data-configuration.png", width=80)
        st.markdown("### ⚙️ KONFIGURASI APLIKASI")

        # File Upload
        uploaded_file = st.file_uploader(
            "📤 Upload Dataset CSV",
            type=["csv"],
            help="Upload file transaksi dengan format CSV"
        )

        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            st.success("✅ File berhasil diupload!")
        else:
            # Use default file
            file_path = "online_retail.csv"
            st.info("ℹ️ Menggunakan dataset default: online_retail.csv")

        st.divider()

        # Clustering Parameters
        st.markdown("### 🎯 PARAMETER ANALISIS")

        n_clusters = st.slider(
            "Jumlah Cluster (k)",
            min_value=2,
            max_value=8,
            value=4,
            help="Tentukan jumlah segmentasi yang diinginkan"
        )

        show_elbow = st.checkbox("Tampilkan Elbow Method", value=True)
        elbow_max_k = st.slider("Max k untuk Elbow", 5, 15, 10) if show_elbow else 10

        st.divider()

        # Analysis Type
        st.markdown("### 🔧 TIPE ANALISIS")
        analysis_type = st.radio(
            "Pilih tipe analisis:",
            ["Full Analysis", "RFM Only", "Clustering Only"]
        )

        st.divider()

        # Run Button
        if st.button("🚀 JALANKAN ANALISIS", type="primary", use_container_width=True):
            st.session_state.run_analysis = True
            st.session_state.analysis_type = analysis_type
        else:
            if 'run_analysis' not in st.session_state:
                st.session_state.run_analysis = False

        st.divider()

        # Info
        with st.expander("📋 Panduan Penggunaan"):
            st.markdown("""
            1. **Upload dataset** CSV Anda (opsional)
            2. **Atur parameter** clustering sesuai kebutuhan
            3. **Klik 'Jalankan Analisis'** untuk memulai
            4. **Lihat hasil** di dashboard utama

            **Format dataset yang didukung:**
            - CustomerID: ID pelanggan
            - InvoiceNo: Nomor invoice
            - InvoiceDate: Tanggal transaksi
            - Quantity: Jumlah barang
            - UnitPrice: Harga satuan
            - Country: Negara (opsional)
            """)

        st.caption("© 2024 - Proyek Akhir Big Data | PySpark MLlib")

    # MAIN CONTENT AREA
    if not st.session_state.get('run_analysis', False):
        # WELCOME SCREEN
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image("https://img.icons8.com/color/300/000000/customer-insight.png", width=300)

            st.markdown("""
            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white; margin: 2rem 0;'>
                <h2 style='color: white;'>Selamat Datang!</h2>
                <p style='font-size: 1.2rem;'>Dashboard Segmentasi Pelanggan menggunakan PySpark MLlib</p>
            </div>
            """, unsafe_allow_html=True)

        # FEATURES OVERVIEW
        st.markdown("### ✨ FITUR UTAMA APLIKASI")

        features = [
            {
                "icon": "📊",
                "title": "RFM Analysis",
                "desc": "Analisis Recency, Frequency, Monetary untuk segmentasi dasar"
            },
            {
                "icon": "🤖",
                "title": "Machine Learning",
                "desc": "Clustering dengan K-Means dan Gaussian Mixture Model"
            },
            {
                "icon": "📈",
                "title": "Visualisasi 3D",
                "desc": "Visualisasi interaktif hasil clustering dalam 3D"
            },
            {
                "icon": "🎯",
                "title": "Business Insights",
                "desc": "Rekomendasi bisnis berdasarkan hasil segmentasi"
            },
            {
                "icon": "📥",
                "title": "Export Results",
                "desc": "Download hasil analisis dalam format CSV"
            },
            {
                "icon": "⚡",
                "title": "Big Data Processing",
                "desc": "Memanfaatkan PySpark untuk processing data besar"
            }
        ]

        cols = st.columns(3)
        for idx, feature in enumerate(features):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>{feature['icon']} {feature['title']}</h3>
                    <p>{feature['desc']}</p>
                </div>
                """, unsafe_allow_html=True)

        return

    # ====================================================
    # RUN ANALYSIS
    # ====================================================

    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Initialize Spark
        status_text.text("🚀 Menginisialisasi Spark Session...")
        progress_bar.progress(5)

        spark = init_spark_session()
        if spark is None:
            st.error("Tidak dapat menginisialisasi Spark. Pastikan Java terinstall.")
            return

        # Load Data
        status_text.text("📥 Memuat dataset...")
        progress_bar.progress(10)

        df = load_data(file_path, spark)
        if df is None:
            st.error("Gagal memuat dataset. Periksa format file.")
            return

        # Preprocessing
        status_text.text("🧹 Membersihkan dan memproses data...")
        progress_bar.progress(20)

        df_clean, cleaning_stats = preprocess_data(df, spark)

        # Display Cleaning Results
        st.markdown("## 📋 HASIL PREPROCESSING DATA")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Data Awal", f"{cleaning_stats['original_count']:,}")
        with col2:
            st.metric("Data Setelah Cleaning", f"{cleaning_stats['cleaned_count']:,}")
        with col3:
            st.metric("Data Dihapus", f"{cleaning_stats['removed_count']:,}")
        with col4:
            st.metric("Retention Rate", f"{cleaning_stats['retention_rate']:.1f}%")

        # Show Sample Data
        with st.expander("👁️ Lihat Sample Data (10 baris pertama)"):
            st.dataframe(df_clean.limit(10).toPandas())

        # RFM Analysis
        if st.session_state.analysis_type in ["Full Analysis", "RFM Only"]:
            status_text.text("💰 Menghitung metrik RFM...")
            progress_bar.progress(40)

            rfm_df, max_date = calculate_rfm(df_clean)
            rfm_pd = rfm_df.toPandas()

            st.markdown("## 📊 ANALISIS RFM (RECENCY, FREQUENCY, MONETARY)")
            st.info(f"📅 **Tanggal referensi (snapshot date):** {max_date}")

            # RFM Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Pelanggan", f"{len(rfm_pd):,}")
            with col2:
                st.metric("Rata-rata Recency", f"{rfm_pd['Recency'].mean():.1f} hari")
            with col3:
                st.metric("Rata-rata Frequency", f"{rfm_pd['Frequency'].mean():.1f}")
            with col4:
                st.metric("Rata-rata Monetary", f"${rfm_pd['Monetary'].mean():,.2f}")

            # RFM Distributions
            st.markdown("### 📈 Distribusi RFM Metrics")

            tab1, tab2 = st.tabs(["Matplotlib", "Plotly"])

            with tab1:
                fig = plot_rfm_distributions(rfm_pd)
                st.pyplot(fig)

            with tab2:
                # Interactive Plotly charts
                fig1 = px.histogram(rfm_pd, x='Recency', nbins=50,
                                    title='Distribusi Recency',
                                    labels={'Recency': 'Hari sejak transaksi terakhir'},
                                    color_discrete_sequence=['#3B82F6'])
                st.plotly_chart(fig1, use_container_width=True)

                fig2 = px.scatter(rfm_pd, x='Recency', y='Frequency',
                                  color='Monetary', size='Monetary',
                                  title='Recency vs Frequency (Ukuran = Monetary)',
                                  labels={'Recency': 'Recency', 'Frequency': 'Frequency'},
                                  hover_data=['CustomerID'])
                st.plotly_chart(fig2, use_container_width=True)

        # Clustering Analysis
        if st.session_state.analysis_type in ["Full Analysis", "Clustering Only"]:
            # Elbow Method
            if show_elbow:
                status_text.text("📈 Menghitung Elbow Method...")
                progress_bar.progress(50)

                # Prepare data for elbow method
                assembler = VectorAssembler(
                    inputCols=["Recency", "Frequency", "Monetary"],
                    outputCol="features_vec"
                )
                df_vector = assembler.transform(rfm_df)

                scaler = StandardScaler(
                    inputCol="features_vec",
                    outputCol="features",
                    withStd=True,
                    withMean=True
                )
                scaler_model = scaler.fit(df_vector)
                final_data = scaler_model.transform(df_vector)

                errors = calculate_elbow_method(final_data, elbow_max_k)

                st.markdown("## 📉 ELBOW METHOD ANALYSIS")
                st.info("Elbow method membantu menentukan jumlah cluster optimal dengan melihat penurunan WSSSE")

                fig = plot_elbow_curve(errors)
                st.pyplot(fig)

                # Display elbow results in table
                elbow_df = pd.DataFrame(errors, columns=['k', 'WSSSE'])
                elbow_df['Reduction'] = elbow_df['WSSSE'].pct_change() * -100
                elbow_df['Reduction'] = elbow_df['Reduction'].fillna(0).round(2)

                st.dataframe(elbow_df.style.background_gradient(subset=['Reduction'], cmap='YlOrRd'))

            # Perform Clustering
            status_text.text("🤖 Melakukan clustering...")
            progress_bar.progress(70)

            clustering_results = perform_clustering(rfm_df, n_clusters)

            st.markdown("## 🎯 HASIL CLUSTERING")

            # Display Model Comparison
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Terbaik", clustering_results['best_model'])
            with col2:
                st.metric("Silhouette Score", f"{clustering_results['best_score']:.4f}")
            with col3:
                st.metric("Jumlah Cluster", n_clusters)

            # Model Scores
            st.info(f"""
            **Perbandingan Model:**
            - **K-Means:** {clustering_results['scores']['kmeans']:.4f}
            - **GMM:** {clustering_results['scores']['gmm']:.4f}

            **Model terbaik:** {clustering_results['best_model']} dengan score {clustering_results['best_score']:.4f}
            """)

            # Cluster Analysis
            status_text.text("🔍 Menganalisis hasil clustering...")
            progress_bar.progress(85)

            predictions_pd, cluster_stats = analyze_clusters(clustering_results['best_predictions'])

            st.markdown("### 📋 PROFIL DETAIL SETIAP CLUSTER")

            # Display cluster statistics
            display_stats = cluster_stats.copy()
            display_stats.columns = [col.replace('_', ' ').title() for col in display_stats.columns]
            st.dataframe(display_stats.style.background_gradient(subset=['Customerid Count'], cmap='Blues'))

            # Display each cluster with interpretation
            st.markdown("### 🏷️ INTERPRETASI SEGMENT PELANGGAN")

            for idx, row in cluster_stats.iterrows():
                with st.container():
                    # Use appropriate CSS class based on cluster type
                    st.markdown(f"""
                    <div class='{row["Color_Class"]}'>
                        <h3>{row['Segment_Name']}</h3>
                        <p><strong>{row['Description']}</strong></p>
                        <p>👥 <strong>{int(row['CustomerID_count'])}</strong> pelanggan | 
                        📅 Rata Recency: <strong>{row['Recency_mean']} hari</strong> | 
                        🔄 Frekuensi: <strong>{row['Frequency_mean']}</strong> | 
                        💰 Monetary: <strong>${row['Monetary_mean']:,.2f}</strong></p>
                        <p>📊 Total Transaksi: {row['TotalTransactions_mean']:.1f} | 
                        💵 Rata Transaksi: ${row['AvgTransactionValue_mean']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Visualizations
            status_text.text("📈 Membuat visualisasi...")
            progress_bar.progress(95)

            st.markdown("### 📊 VISUALISASI HASIL CLUSTERING")

            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["3D Visualization", "Cluster Comparison", "Interactive"])

            with viz_tab1:
                fig_3d = plot_cluster_3d(predictions_pd)
                st.plotly_chart(fig_3d, use_container_width=True)

            with viz_tab2:
                fig_comparison = plot_cluster_comparison(predictions_pd)
                st.pyplot(fig_comparison)

            with viz_tab3:
                # Interactive scatter matrix
                fig_matrix = px.scatter_matrix(
                    predictions_pd,
                    dimensions=['Recency', 'Frequency', 'Monetary'],
                    color='prediction',
                    title='Scatter Matrix of RFM Features by Cluster',
                    labels={col: col for col in predictions_pd.columns},
                    hover_data=['CustomerID']
                )
                st.plotly_chart(fig_matrix, use_container_width=True)

                # Parallel coordinates plot
                fig_parallel = px.parallel_coordinates(
                    predictions_pd,
                    color='prediction',
                    dimensions=['Recency', 'Frequency', 'Monetary'],
                    color_continuous_scale=px.colors.diverging.Tealrose,
                    title='Parallel Coordinates Plot'
                )
                st.plotly_chart(fig_parallel, use_container_width=True)

        # Generate Recommendations
        status_text.text("💡 Membuat rekomendasi bisnis...")
        progress_bar.progress(98)

        if 'cluster_stats' in locals():
            recommendations = generate_recommendations(cluster_stats)

            st.markdown("## 🎯 REKOMENDASI STRATEGI MARKETING")

            for segment, recs in recommendations.items():
                with st.expander(f"{segment} - Action Plan"):
                    for i, rec in enumerate(recs, 1):
                        st.markdown(f"{i}. {rec}")

            # Business Impact Summary
            st.markdown("### 📈 ESTIMASI DAMPAK BISNIS")

            impact_cols = st.columns(3)
            with impact_cols[0]:
                st.metric("Target Pelanggan", f"{len(predictions_pd):,}")
            with impact_cols[1]:
                vip_count = cluster_stats[cluster_stats['Segment_Name'].str.contains('VIP')]['CustomerID_count'].sum()
                st.metric("VIP Customers", f"{vip_count:,}")
            with impact_cols[2]:
                total_monetary = cluster_stats['Monetary_mean'] * cluster_stats['CustomerID_count']
                st.metric("Total Revenue Potential", f"${total_monetary.sum():,.0f}")

        # Finish
        progress_bar.progress(100)
        status_text.text("✅ Analisis selesai!")

        # Download Results
        st.markdown("## 📥 DOWNLOAD HASIL ANALISIS")

        if 'predictions_pd' in locals():
            # Prepare data for download
            download_data = predictions_pd.copy()
            download_data['Segment'] = download_data['prediction'].apply(
                lambda x: cluster_stats.loc[x, 'Segment_Name'] if x in cluster_stats.index else f"Cluster {x}"
            )

            csv_data = download_data.to_csv(index=False)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📊 Download Data Segmentasi (CSV)",
                    data=csv_data,
                    file_name="customer_segmentation_results.csv",
                    mime="text/csv",
                    help="Download hasil segmentasi untuk analisis lebih lanjut"
                )

            with col2:
                # Summary report
                summary_report = f"""
                LAPORAN SEGMENTASI PELANGGAN
                =============================

                Tanggal Analisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                Total Pelanggan: {len(predictions_pd):,}
                Jumlah Cluster: {n_clusters}
                Model Terbaik: {clustering_results['best_model']}
                Silhouette Score: {clustering_results['best_score']:.4f}

                SUMMARY PER CLUSTER:
                --------------------
                """

                for idx, row in cluster_stats.iterrows():
                    summary_report += f"""
                {row['Segment_Name']}:
                  - Jumlah: {int(row['CustomerID_count']):,} pelanggan
                  - Rata Recency: {row['Recency_mean']:.1f} hari
                  - Rata Frequency: {row['Frequency_mean']:.1f}
                  - Rata Monetary: ${row['Monetary_mean']:.2f}
                  - Total Revenue: ${row['Monetary_mean'] * row['CustomerID_count']:,.0f}
                    """

                st.download_button(
                    label="📋 Download Summary Report (TXT)",
                    data=summary_report,
                    file_name="segmentation_summary.txt",
                    mime="text/plain"
                )

        # Success Message
        st.balloons()
        st.success("🎉 **ANALISIS SEGMENTASI PELANGGAN TELAH BERHASIL DISELESAIKAN!**")

        # Next Steps
        with st.expander("🔄 Langkah Selanjutnya"):
            st.markdown("""
            1. **Implementasi Campaign:** Gunakan hasil segmentasi untuk campaign marketing
            2. **Monitor Hasil:** Track perubahan segmentasi setiap bulan
            3. **Optimasi Model:** Update model dengan data terbaru
            4. **A/B Testing:** Test strategi berbeda untuk tiap segment
            5. **Integration:** Integrasi dengan CRM atau marketing automation
            """)

    except Exception as e:
        st.error(f"❌ **ERROR:** {str(e)}")
        st.error("Terjadi kesalahan dalam proses analisis.")

        with st.expander("🔧 Troubleshooting Tips"):
            st.markdown("""
            **Kemungkinan penyebab error:**
            1. **Format dataset tidak sesuai** - Pastikan kolom yang diperlukan ada
            2. **Memory tidak cukup** - Kurangi ukuran dataset atau tambah memory Spark
            3. **Java not installed** - Install Java 8 atau 11
            4. **File corrupt** - Periksa integritas file CSV
            5. **Spark configuration** - Pastikan Spark terinstall dengan benar

            **Solusi:**
            - Cek format dataset Anda
            - Kurangi jumlah data untuk testing
            - Restart aplikasi dan coba lagi
            - Pastikan Java sudah terinstall
            """)

        # Reset analysis state
        if 'run_analysis' in st.session_state:
            st.session_state.run_analysis = False


# ====================================================
# RUN THE APPLICATION
# ====================================================

if __name__ == "__main__":
    main()


