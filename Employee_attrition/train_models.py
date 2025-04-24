# train_models.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle

# Load data with error handling
try:
    df = pd.read_csv(r'C:\Users\snbal\Downloads\Emp_attrition.csv')
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# ================== ATTRITION MODEL ==================
def train_attrition_model():
    print("\nTraining Attrition Model...")
    
    # Feature selection
    x = df[['OverTime', 'Age', 'MonthlyIncome', 'DailyRate', 'TotalWorkingYears', 'MonthlyRate']]
    y = df['Attrition']
    
    # Encoding
    overtime_encoder = LabelEncoder()
    x['OverTime'] = overtime_encoder.fit_transform(x['OverTime'])
    
    y_encoder = LabelEncoder() if y.dtype == 'object' else None
    if y_encoder:
        y = y_encoder.fit_transform(y)
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }
    
    # Model training with GridSearchCV
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(x_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")
    print(f"Validation accuracy: {grid_search.best_score_:.2f}")
    
    # Save artifacts
    artifacts = {
        'model': best_model,
        'overtime_encoder': overtime_encoder,
        'target_encoder': y_encoder,
        'feature_names': x.columns.tolist(),
        'test_data': (x_test, y_test)  # Optional: for later evaluation
    }
    
    with open('attrition_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("Attrition model saved to 'attrition_model.pkl'")

# ================== PERFORMANCE MODEL ==================
def train_performance_model():
    print("\nTraining Performance Model...")
    
    # Feature selection
    x = df[['Age', 'TotalWorkingYears', 'MonthlyIncome', 'PercentSalaryHike', 'JobRole']]
    y = df['PerformanceRating']
    
    # Encoding
    jobrole_encoder = LabelEncoder()
    x['JobRole'] = jobrole_encoder.fit_transform(x['JobRole'])
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    
    # Model training
    model = LogisticRegression(max_iter=1000, random_state=2)
    model.fit(x_train, y_train)
    
    # Save artifacts
    artifacts = {
        'model': model,
        'jobrole_encoder': jobrole_encoder,
        'valid_job_roles': df['JobRole'].unique().tolist(),
        'feature_names': x.columns.tolist(),
        'test_data': (x_test, y_test)  # Optional: for later evaluation
    }
    
    with open('performance_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("Performance model saved to 'performance_model.pkl'")

# ================== MAIN EXECUTION ==================
if __name__ == "__main__":
    train_attrition_model()
    train_performance_model()
    
    print("\nAll models trained and saved successfully!")