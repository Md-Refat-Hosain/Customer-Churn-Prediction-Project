
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
<img width="1352" height="202" alt="Screenshot 2025-07-23 at 2 54 28 PM" src="https://github.com/user-attachments/assets/b7bc0931-866a-4b6b-853d-25687f5c415b" />

### insights from various categorical feature vs churn
<img width="3780" height="1890" alt="Untitled design" src="https://github.com/user-attachments/assets/b400d50b-34aa-4f03-8fe6-4277a3f6ce82" />
<img width="3780" height="1890" alt="Untitled design (1)" src="https://github.com/user-attachments/assets/52ab368c-d78f-4722-847c-6636432a2dec" />
<img width="1354" height="253" alt="Screenshot 2025-07-23 at 4 18 43 PM" src="https://github.com/user-attachments/assets/8a4a957f-f254-45f3-a121-e07c39578b66" />

### insights from various contineous feature vs churn



<img width="3780" height="1890" alt="Untitled design" src="https://github.com/user-attachments/assets/9d90e0cd-04ad-4931-9c64-53e7c617bdbd" />
<img width="1238" height="165" alt="Screenshot 2025-07-23 at 4 31 37 PM" src="https://github.com/user-attachments/assets/0ee78f76-cc88-4c54-a600-233872a419a6" />

### Problematic multicolinearity Detection Among Independent Features
<img width="3780" height="1890" alt="Untitled design" src="https://github.com/user-attachments/assets/0891578d-f083-48e8-af71-aba67f1feeb3" />

From the vif analysis I found out that features which vif value greater than 5 needs to be dropped, But I was not sure which features to drop the I created heatmap to see multicolinearity values and found out which features to be dropped.
* Heat map to be sure which feature to drop
  (Before features drop)
<img width="1314" height="1250" alt="download (1)" src="https://github.com/user-attachments/assets/8ee33c40-cc02-49d9-91de-c5b24ac54062" />











