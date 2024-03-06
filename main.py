import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Function to load train images from a folder and assign labels
def load_train_images_from_folder(folder, target_shape=None):
    images = []
    labels = []
    for emotion_folder in os.listdir(folder):
        emotion_folder_path = os.path.join(folder, emotion_folder)
        if os.path.isdir(emotion_folder_path):
            for filename in os.listdir(emotion_folder_path):
                if filename.endswith('.jpg'):  # Assuming images are in JPEG format
                    img = cv2.imread(os.path.join(emotion_folder_path, filename))
                    if img is not None:
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                        images.append(img)
                        labels.append(emotion_folder)  # Assign label based on folder name
                    else:
                        print(f"Warning: Unable to load {filename}")
    return images, labels

# Function to load test images from a folder and assign labels
def load_test_images_from_folder(folder, target_shape=None):
    images = []
    labels = []
    for emotion_folder in os.listdir(folder):
        emotion_folder_path = os.path.join(folder, emotion_folder)
        if os.path.isdir(emotion_folder_path):
            for filename in os.listdir(emotion_folder_path):
                if filename.endswith('.jpg'):  # Assuming images are in JPEG format
                    img = cv2.imread(os.path.join(emotion_folder_path, filename))
                    if img is not None:
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                        images.append(img)
                        labels.append(emotion_folder)  # Assign label based on folder name
                    else:
                        print(f"Warning: Unable to load {filename}")
    return images, labels

# Function to plot images grouped by clusters
def plot_cluster_images(images, cluster_assignments, titles):
    n_clusters = len(images)
    plt.figure(figsize=(15, 5))
    for cluster in range(n_clusters):
        cluster_images = images[cluster]
        cluster_title = f"Cluster {cluster} ({len(cluster_images)} images)"
        for i, image in enumerate(cluster_images):
            plt.subplot(n_clusters, len(cluster_images), i + 1 + cluster * len(cluster_images))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_idx = np.where(cluster_assignments == cluster)[0][i] # Find the indices for the current cluster
            plt.title(f"{titles[image_idx]}")
            plt.axis('off')
    plt.show()

# Function to calculate confusion matrix
def calculate_confusion_matrix(test_labels, predicted_labels, labels):
    return confusion_matrix(test_labels, predicted_labels, labels=labels)

# Folder paths
data_folder = './emotions_dataset'  # Folder with training data
test_folder = './test_faces'  # Folder with test data

# Load images and labels from the 'emotions_dataset' folder and resize them to (200, 200)
images, labels = load_train_images_from_folder(data_folder, target_shape=(200, 200))

# Apply K-Means clustering
n_clusters = len(set(labels))  # Automatically detect the number of clusters based on unique labels
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# Reshape the images and convert them to grayscale
image_data = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten() for image in images]

# Convert the list of 1D arrays to a 2D numpy array
image_data = np.array(image_data)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(image_data)

# Train the K-Means model
kmeans.fit(scaled_data)

# Get cluster assignments
cluster_assignments = kmeans.labels_

# Group images by clusters
cluster_images = [[] for _ in range(n_clusters)]
for i, label in enumerate(labels):
    cluster = cluster_assignments[i]
    cluster_images[cluster].append(images[i])

# Assign titles for images in each cluster
titles = labels

# Plot images grouped by clusters
plot_cluster_images(cluster_images, cluster_assignments, titles)

### TESTING ###
# Emotion labels corresponding to numerical predictions
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load test images from the 'test_faces' folder
test_images, test_labels = load_test_images_from_folder(test_folder, target_shape=(200, 200))

# Predict emotions for test images and visualize results
predicted_labels = [emotion_labels[kmeans.predict(scaler.transform([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten() for image in [test_image]])).item()] for test_image in test_images]

# Debugging statements
print("Test Labels:", test_labels)
print("Predicted Labels:", predicted_labels)

# Calculate confusion matrix
confusion_mat = confusion_matrix(test_labels, predicted_labels, labels=emotion_labels)
print("Confusion Matrix:")
print(confusion_mat)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels, average='weighted')
recall = recall_score(test_labels, predicted_labels, average='weighted')

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.get_cmap('Blues'))
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(emotion_labels))
plt.xticks(tick_marks, emotion_labels, rotation=45)
plt.yticks(tick_marks, emotion_labels)
plt.xlabel('Predicted Emotion')
plt.ylabel('True Emotion')
plt.show()