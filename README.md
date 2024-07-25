# Machine Learning Pipeline for Customer Churn Prediction

This project focuses on developing a comprehensive machine learning pipeline using Scikit-Learn to predict customer churn. The pipeline is designed to streamline the preprocessing and modeling stages to enhance efficiency and performance.

## Dataset

The dataset used in this project is the [Churn Modelling dataset](https://www.kaggle.com/datasets/aakash50897/churn-modellingcsv/data) from Kaggle, which contains information about customer behavior and churn status.

## Project Overview

The main components of the pipeline include:
- **Data Imputation:** Handling missing values to ensure data completeness.
- **Feature Scaling:** Normalizing data to improve model convergence and performance.
- **PCA (Principal Component Analysis):** Reducing dimensionality while retaining significant variance.
- **One-Hot Encoding:** Converting categorical variables into a machine-readable format.
- **Model Fine-Tuning:** Optimizing the estimator to achieve the best predictive performance.

## Steps

1. **Data Loading and Inspection:**
   - Load the dataset.
   - Inspect the dataset and drop irrelevant columns.
   - Split the dataset into training and testing sets.

2. **Pipeline Construction:**
   - Build separate pipelines for numerical and categorical data processing.
   - Numerical pipeline includes imputation, feature scaling, and PCA.
   - Categorical pipeline includes imputation and one-hot encoding.

3. **Model Training and Evaluation:**
   - Integrate the pipelines into a unified preprocessing workflow.
   - Train and fine-tune a Random Forest classifier to predict customer churn.
   - Evaluate the model performance using appropriate metrics.

## Key Features

- The pipeline is highly adaptable and modular, allowing for easy integration of different datasets and models.
- Scikit-Learn's Pipeline and ColumnTransformer classes are utilized to create an efficient and scalable machine learning workflow.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-Learn

## Installation

Clone the repository:
```sh
git clone https://github.com/Manuel-ventura/ML-Pipeline-Scikit-Learn/tree/main
cd ML-Pipeline-Scikit-Learn
```

Install the required packages:
```sh
pip install -r requirements.txt
```

## Usage

Run the Jupyter Notebook to see the pipeline in action:
```sh
jupyter notebook ML-Pipeline-Scikit-Learn.ipynb
```

## Future Work

- **Model Enhancement:** Implementing cross-validation, hyperparameter tuning, and ensemble methods to improve model accuracy.
- **Deployment:** Developing an API for prediction and deploying the model using cloud platforms.
- **Extending the Project:** Adding more datasets and experimenting with different machine learning algorithms.

## Contributions

Feel free to fork this repository, submit issues and pull requests. Your contributions are greatly appreciated.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
