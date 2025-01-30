# Credit Scoring & Risk Assessment - Machine Learning Project

##  Project Overview
This project focuses on credit scoring and risk assessment using machine learning. The objective is to predict whether a customer will default on a loan based on financial and demographic features. The model is trained using a classification algorithm and evaluated for accuracy and reliability.

## Dataset Information
- Dataset Name: `bank.csv`
- Source: (Provide dataset source if applicable)
- Size: (Number of rows and columns)
- Features: Includes attributes like age, income, loan amount, credit history, etc.
- Target Variable: Loan Default (`yes/no` or `1/0`)

## Technology Stack
- Programming Language: Python
- Libraries Used:
  - `pandas` (Data handling)
  - `numpy` (Numerical operations)
  - `scikit-learn` (Machine Learning)
  - `matplotlib` & `seaborn` (Visualization)
  - `FastAPI` (Model Deployment)
  - `pickle` (Model Serialization)

##  Project Workflow
### 1 Data Preprocessing
- Handling missing values (median imputation)
- Encoding categorical variables (Label Encoding, One-Hot Encoding)
- Feature scaling using `StandardScaler`

### 2 Model Selection & Training
- Used Random Forest Classifier for prediction
- Train-test split (80% training, 20% testing)
- Hyperparameter tuning with `GridSearchCV`

### 3 Model Evaluation
- Performance metrics:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix

### 4 Model Deployment
- Deployed as an API using **FastAPI**
- Model receives input features and returns a prediction
- API can be tested using Postman or `curl`

##  How to Run the Project
### 1 Clone the Repository
```bash
git clone https://github.com/kalkidanzenebe/credit-scoring-ml.git
cd credit-scoring-ml
```

### 2 Install Dependencies
```bash
pip install -r requirements.txt
```

### 3 Run the ML Model
```bash
python main.py
```

### 4 Deploy the API 
```bash
uvicorn app:main --reload
```

## Results & Future Improvements
- Achieved Accuracy: (Add model accuracy score)
- Potential Improvements:
  - Try deep learning models (e.g., Neural Networks)
  - Feature engineering for better predictions
  - More extensive hyperparameter tuning

## License
This project is for educational purposes only.




