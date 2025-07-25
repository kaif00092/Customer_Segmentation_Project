Customer Segmentation using K-Means Clustering 🛍️
This project uses the K-Means clustering algorithm to segment customers into different groups based on their annual income and spending score. This is a common technique in marketing to understand customer behavior and enable targeted advertising.

🎯 Project Goal
The main goal is to identify distinct customer groups to help businesses create more effective marketing strategies. The project performs the following steps:

Uses a sample customer dataset.

Applies the Elbow Method to determine the optimal number of clusters.

Runs the K-Means algorithm to create the customer segments.

Visualizes the final clusters in a scatter plot for easy interpretation.

🛠️ How to Run
To run this project on your local machine, follow these steps.

1. Prerequisites
Make sure you have Python installed.


2. Install Required Libraries
You can install all the necessary libraries using pip:

Bash

pip install pandas scikit-learn matplotlib seaborn
3. Execute the Script
Run the main Python script from your terminal:

Bash

python segment.py
📈 Expected Output
After running the script, two image files will be saved in your project folder:

elbow_method_plot.png: This graph helps you find the optimal number of clusters (in this case, 5).

customer_segments.png: This scatter plot shows the final customer segments, each marked with a different color.
