import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

st.set_page_config(page_title="Student Lifestyle Clustering", layout="wide")
st.title("🎓 Student Lifestyle Clustering App")

st.markdown("""
This app groups students into **lifestyle clusters** based on:
- Study hours
- Social activity
- Exercise
- Sleep patterns
- Cafeteria spending

It uses **K-Means Clustering** to find natural groupings and helps universities design better student wellness programs.
""")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("🔍 Sample of Uploaded Data")
    st.dataframe(data.head())

    # Preprocessing
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

    features = ['Age', 'Gender', 'Study Hours per Week',
                'Exercise Frequency (per week)', 'Sleep Hours per Night',
                'Cafeteria Spend ($/week)', 'Social Activity Score (1–100)']

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])

    # Sidebar for cluster selection
    st.sidebar.header("⚙️ Clustering Options")
    num_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, 4)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    data['Cluster'] = clusters

    st.subheader("📊 Clustered Dataset")
    st.dataframe(data[['Cluster'] + features])

    st.subheader("📈 Cluster Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(data['Study Hours per Week'], 
                         data['Social Activity Score (1–100)'],
                         c=data['Cluster'], cmap='Set1', s=100, alpha=0.8, edgecolors='k')
    ax.set_xlabel('Study Hours per Week')
    ax.set_ylabel('Social Activity Score')
    ax.set_title('Student Lifestyle Clusters')
    st.pyplot(fig)

    st.markdown("""
    ### 📌 How to Use This App
    1. Upload your student lifestyle data CSV
    2. Select number of clusters in sidebar
    3. Explore the grouped student behaviors
    """)
else:
    st.warning("Please upload a CSV file to continue.")
