import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def build_linear_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def make_predictions(model, X):
    return model.predict(X)

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

def perform_kmeans(X, clusters):
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init="auto")
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_, kmeans.inertia_

def clustering_scores(X, labels):
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return sil, ch, db

def evaluate_multiple_k(X, k_values):
    sil_vals, ch_vals, db_vals = [], [], []
    for k in k_values:
        labels, _, _ = perform_kmeans(X, k)
        unique_labels = len(set(labels))
        # Silhouette score requires 2 <= n_labels <= n_samples - 1
        if 2 <= unique_labels <= len(X) - 1:
            sil, ch, db = clustering_scores(X, labels)
        else:
            sil, ch, db = None, None, None
        sil_vals.append(sil)
        ch_vals.append(ch)
        db_vals.append(db)
    return sil_vals, ch_vals, db_vals

def elbow_method(X, k_values):
    distortions = []
    for k in k_values:
        _, _, inertia = perform_kmeans(X, k)
        distortions.append(inertia)
    return distortions

if __name__ == "__main__":
    # A1 & A2: Single feature regression
    X_train1 = np.array([[1], [2], [3], [4], [5]])
    y_train1 = np.array([10, 20, 30, 40, 50])
    X_test1 = np.array([[6], [7]])
    y_test1 = np.array([60, 70])

    model1 = build_linear_model(X_train1, y_train1)
    train_pred1 = make_predictions(model1, X_train1)
    test_pred1 = make_predictions(model1, X_test1)
    print("A1 & A2 - Train Metrics:", regression_metrics(y_train1, train_pred1))
    print("A1 & A2 - Test Metrics:", regression_metrics(y_test1, test_pred1))

    # A3: Multiple feature regression
    X_train2 = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
    y_train2 = np.array([11, 12, 13, 14, 15])
    X_test2 = np.array([[6, 0], [7, -1]])
    y_test2 = np.array([16, 17])

    model2 = build_linear_model(X_train2, y_train2)
    train_pred2 = make_predictions(model2, X_train2)
    test_pred2 = make_predictions(model2, X_test2)
    print("A3 - Train Metrics:", regression_metrics(y_train2, train_pred2))
    print("A3 - Test Metrics:", regression_metrics(y_test2, test_pred2))

    # A4 & A5: KMeans clustering and evaluation (k=2)
    data_cluster = np.array([[1.0, 5.0], [2.0, 4.5], [1.5, 3.5], [5.0, 1.0], [6.0, 2.5], [6.5, 0.5]])
    labels_2, centers_2, _ = perform_kmeans(data_cluster, 2)
    print("A4 - Cluster Labels:", labels_2)
    print("A4 - Cluster Centers:\n", centers_2)
    print("A5 - Evaluation Scores:", clustering_scores(data_cluster, labels_2))

    # A6: Evaluate clustering metrics across multiple k values (2 to n_samples - 1)
    k_values = range(2, len(data_cluster))  # max clusters < number of samples
    sil_scores, ch_scores, db_scores = evaluate_multiple_k(data_cluster, k_values)

    plt.figure()
    plt.plot(k_values, sil_scores, marker='o', label='Silhouette Score')
    plt.plot(k_values, ch_scores, marker='x', label='Calinski-Harabasz Score')
    plt.plot(k_values, db_scores, marker='s', label='Davies-Bouldin Index')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Score')
    plt.title('A6 - Clustering Evaluation Metrics vs. k')
    plt.legend()
    plt.grid(True)
    plt.show()

    # A7: Elbow method for optimal k (k values capped to avoid > n_samples)
    k_elbow = range(2, len(data_cluster) + 1)  # max clusters = number of samples
    distortions = elbow_method(data_cluster, k_elbow)

    plt.figure()
    plt.plot(k_elbow, distortions, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion (Inertia)')
    plt.title('A7 - Elbow Method Plot')
    plt.grid(True)
    plt.show()

