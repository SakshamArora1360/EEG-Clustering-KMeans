# EEG Clustering with K-Means

This project applies **K-Means clustering** on EEG brainwave data to identify meaningful groups based on cognitive states such as **Attention** and **Mediation**.  
The workflow includes **data preprocessing, scaling, clustering, and visualization**.

---

## 📊 Features
- Reads and processes EEG dataset (`EEG_data.csv`).
- Drops irrelevant columns to focus on important features.
- Standardizes data using **StandardScaler**.
- Determines optimal cluster count using:
  - **Elbow Method** (Inertia)
  - **Silhouette Score**
- Applies **K-Means clustering**.
- Visualizes clusters of `Attention` vs `Mediation` with centroids.

---

## 📂 Project Structure
EEG-Clustering-KMeans/
│── EEG_data.csv # Dataset 
│── eeg_kmeans_clustering.py # Main Python script
│── README.md # Project documentation

---

## Output Example
- Elbow and Silhouette Score Plots
- <img width="704" height="367" alt="image" src="https://github.com/user-attachments/assets/0f846f30-cb23-41c2-91b2-cb5512917de6" />
- Cluster Scatter Plots with Centroids
- <img width="376" height="287" alt="image" src="https://github.com/user-attachments/assets/c5cc6004-9ce5-4fb2-8857-a281f9c3aa58" />

  ---
  
## 📈 Results
- Elbow Plot helps visualize the drop in inertia to choose possible K.
- Silhouette Score Plot identifies the optimal K.
- Final clusters are plotted on Attention vs Mediation, showing distinct groups with centroids.

---

## 🧠 Use Case
This project is useful for:
- Understanding clustering in EEG data.
- Exploring brainwave patterns related to cognitive states.
- Learning practical applications of unsupervised machine learning.

---

## 📌 Example Visualization
- Clustered scatter plot of Attention vs Mediation with centroids.
- Helps in interpreting cognitive state separation.
