
#  Telecom Customer Churn Prediction Project


This is a end-to-end data science project showcasing the complete process of creating a machine learning model for predicting customer churn prediction , covering everything from managing untidy raw data to developing a reliable model to deploy via FastAPI.

## 🎯 Overview & Goal


This project is all about helping a telecom company keep its customers.

### The Problem: 
Customers often leave phone companies. When they leave, the company loses money.

### Our Goal:
To analyse and build a machine learning model that can predict which customers are likely to leave.

### How We Did It:

* Explored Data: Looked closely at customer info to understand why people leave.

* Cleaned Data: Prepared the data so the computer can understand it.

* Built a Model: Used a powerful technique called Random Forest and fine-tuned it to be as accurate as possible.

* Evaluated Results: Checked how well our model predicts churn.


### Why it Matters: 

By finding "at-risk" customers early, the company can try to keep them, saving money and making customers happier!

### Tools Used: 
Python (Pandas, Scikit-learn, Matplotlib, Seaborn)

# 🚀 Project Structure & How to Explore

To understand and explore this project, navigate through the following key sections and files in the repository:

### 1. Data & Notebooks

* **`data/`**: This directory contains the datasets used in the project.
    * `data/raw/`: Stores the original, untouched dataset.
    * `data/processed/`: Contains the cleaned and preprocessed data ready for modeling.
* **`notebooks/`**: This folder houses the Jupyter Notebooks detailing each stage of the project.
    * `01_EDA.ipynb`: Comprehensive Exploratory Data Analysis, including all visualizations and initial insights.
    * `02_Preprocessing.ipynb`: Detailed steps for data cleaning, feature engineering, and transformation.
    * `03_Model_Training_Evaluation.ipynb`: Model selection, initial training, and performance evaluation.
    * `04_Hyperparameter_Tuning.ipynb`: All the model used  hyperparameter optimization to get the best model's hyperparameters.


### 2. Model Artifacts

* **`models/`**: This directory stores the trained machine learning models.
    * `lrc_bset_model.joblib`: The final, best-performing logistic regression model saved for future use.


### 3. Supporting Files

* **`requirements.txt`**: Lists all the Python libraries and their exact versions required to run this project, ensuring environment reproducibility.
* **`images/`**: Contains all the plots and screenshots embedded in this `README.md` for visual context.



## ✨ Key Insights

This section presents the key findings and visualizations from the Exploratory Data Analysis (EDA) that informed the modeling approach.

### insights from various categorical features
<img width="3780" height="1890" alt="Untitled design" src="https://github.com/user-attachments/assets/a9a440d4-f6d2-4a35-8bae-05b9a130b2f6" />
<img width="3780" height="1890" alt="Untitled design (1)" src="https://github.com/user-attachments/assets/02e4b152-dc2c-46a5-9adf-2af6d51f0cc3" />

