# Required libraries ko import karein
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Project script started...")

# --- Step 1: Create a Dummy Dataset ---

data = {
    'AnnualIncome_k$': [15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 23, 23, 24, 25, 28, 28, 29, 30, 33, 33, 34, 37, 38, 39, 39, 39, 40, 40, 42, 43, 44, 46, 47, 48, 48, 49, 50, 54, 54, 54, 54, 57, 58, 59, 60, 60, 60, 61, 62, 62, 62, 63, 63, 64, 65, 65, 67, 67, 69, 70, 71, 71, 72, 73, 73, 74, 75, 76, 76, 77, 78, 78, 78, 78, 79, 81, 85, 86, 87, 87, 88, 93, 97, 98, 99, 101, 103, 103, 113, 120, 126, 137],
    'SpendingScore_1-100': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72, 35, 99, 5, 79, 41, 78, 35, 95, 3, 91, 14, 83, 4, 73, 53, 73, 36, 92, 21, 69, 42, 97, 27, 75, 55, 68, 43, 94, 16, 56, 47, 90, 15, 63, 52, 60, 49, 93, 20, 51, 42, 85, 23, 65, 48, 59, 50, 87, 24, 62, 49, 59, 51, 88, 25, 60, 46, 51, 42, 90, 10, 61, 40, 55, 47, 89, 9, 68, 1, 48, 1, 57, 18, 46, 20, 50, 1, 49, 15, 52, 28, 41, 32, 86, 9, 36]
}
df = pd.DataFrame(data)

print("Dataset created successfully.")
# --- Step 2: Select Features and Scale them ---
# Hum 'Annual Income' aur 'Spending Score' ka use karenge.
features = df[['AnnualIncome_k$', 'SpendingScore_1-100']]

# Scaling zaroori hai taaki koi ek feature model ko dominate na kare.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print("Data has been scaled.")

# --- Step 3: Find the Optimal Number of Clusters (Elbow Method) ---
# Hum dekhenge ki kitne clusters (k) best hain.
wcss = [] # Within-Cluster Sum of Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
# Save the plot to a file
plt.savefig('elbow_method_plot.png')
print("Elbow method plot saved as 'elbow_method_plot.png'.")
# From the plot, we can see the "elbow" is at k=5.

# --- Step 4: Apply K-Means with the Optimal k ---
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
# `fit_predict` model ko train karke har customer ka cluster number dega.
df['Cluster'] = kmeans.fit_predict(scaled_features)

print(f"K-Means applied with {optimal_k} clusters.")

# --- Step 5: Visualize the Clusters ---
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='AnnualIncome_k$', y='SpendingScore_1-100', hue='Cluster', palette='viridis', s=100, legend='full')

# Add centroids to the plot
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
# Save the final cluster plot
plt.savefig('customer_segments.png')
print("Customer segmentation plot saved as 'customer_segments.png'.")
print("\nProject execution finished. Check the folder for output images.")