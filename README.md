# Handwritten Digit Classifier Web App

An interactive machine learning app for classifying handwritten digits using multiple classifiers and evaluating their performance. Originally this was a class assignment, this project was expanded into a web app using with **Scikit-learn**, **Streamlit**, and **Matplotlib** using the classic `digits` dataset for model training and comparison..

## Project Overview

This app compares the performance of four classification algorithms on the digits dataset:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (Linear Kernel)**
- **Support Vector Machine (RBF Kernel)**
- **Naive Bayes (Gaussian)**

The application provides:

- Model training and prediction
- Performance metrics (Accuracy, Precision, Recall)
- Cross-validation scores
- Visualizations of results

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/digit-classifier-app.git
cd digit-classifier-app

## 2. Install Dependencies
pip install -r requirements.txt
streamlit run main.py

### 3. Run the applcation
streamlit run main.py
```
