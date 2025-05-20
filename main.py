import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score


# Load a the dataset 
digits = load_digits()
X = digits.data
y = digits.target

# Sidebar settings
st.sidebar.header("⚙️ Model Settings")
classifier_name = st.sidebar.selectbox("Choose Classifier", ("K-Nearest Neighbors", "Linear SVM", "Gaussian Kernel SVM", "Naive Bayes"))
k_value = st.sidebar.slider("K (for KNN only)", 1, 15, 5)
show_images = st.sidebar.checkbox("Show Sample Images", value=True)

# Show sample digit images
if show_images:
    st.subheader("🖼 Sample Digits")
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(digits.images[i], cmap="gray")
        ax.set_title(f"Label: {digits.target[i]}")
        ax.axis("off")
    st.pyplot(fig)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  Select model
if classifier_name == "K-Nearest Neighbors":
    model = KNeighborsClassifier(n_neighbors= k_value)
elif classifier_name == "Linear SVM":
    model = SVC(kernel='linear')
elif classifier_name == "Gaussian Kernel SVM":
    model = SVC(kernel='rbf')  # RBF = Gaussian kernel
elif classifier_name == "Naive Bayes":
    model = GaussianNB()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display metrics
st.subheader(f"📊 Evaluation Results ({classifier_name})")
st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
st.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.4f}")
st.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")

# Show classification report
st.subheader("📋 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(report)