# train_model.py
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer



def create_features(df):
    """Create new features to improve model performance"""
    df_new = df.copy()
    df_new['AST_ALT_ratio'] = df_new['Aspartate_Aminotransferase'] / (df_new['Alamine_Aminotransferase'] + 1)
    df_new['TB_DB_ratio'] = df_new['Total_Bilirubin'] / (df_new['Direct_Bilirubin'] + 0.001)
    df_new['AP_ALT_ratio'] = df_new['Alkaline_Phosphotase'] / (df_new['Alamine_Aminotransferase'] + 1)
    df_new['Age_ALT'] = df_new['Age'] * df_new['Alamine_Aminotransferase'] / 100
    df_new['Age_AST'] = df_new['Age'] * df_new['Aspartate_Aminotransferase'] / 100
    df_new['ALT_squared'] = df_new['Alamine_Aminotransferase'] ** 2
    df_new['AST_squared'] = df_new['Aspartate_Aminotransferase'] ** 2
    df_new['Liver_Function_Score'] = df_new['Albumin'] / (df_new['Total_Bilirubin'] + 0.001)
    df_new['Gender_ALT'] = df_new['Gender'] * df_new['Alamine_Aminotransferase']
    return df_new

def train_and_save_model():
    try:
        df = pd.read_csv(r'E:\Proj_4\venv\indian_liver_patient - indian_liver_patient (1).csv')
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # Preprocessing
    gender_encoder = LabelEncoder()
    df['Gender'] = gender_encoder.fit_transform(df['Gender'])
    df_enhanced = create_features(df)

    gender_categories = dict(zip(
        gender_encoder.classes_,
        gender_encoder.transform(gender_encoder.classes_)
    ))

    X = df_enhanced.drop('Dataset', axis=1)
    y = df_enhanced['Dataset'].replace({2: 0})

    # Imputation
    imputer = KNNImputer(n_neighbors=5)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance
    try:
        smote_tomek = SMOTETomek(random_state=42)
        X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"SMOTETomek error: {e}")

    # Power Transform
    power_transformer = PowerTransformer(method='yeo-johnson')
    X_train_transformed = pd.DataFrame(
        power_transformer.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_transformed = pd.DataFrame(
        power_transformer.transform(X_test),
        columns=X_test.columns
    )

    # Feature selection
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = SelectFromModel(base_model, threshold='median')
    selector.fit(X_train_transformed, y_train)

    X_train_selected = selector.transform(X_train_transformed)
    X_test_selected = selector.transform(X_test_transformed)

    # Hyperparameter tuning
    rf_params = {
        'n_estimators': [200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_samples': [0.7, 0.9]
    }

    rf_model = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        rf_params,
        cv=StratifiedKFold(5),
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    rf_model.fit(X_train_selected, y_train)

    # Save all components
    joblib.dump({
        'model': rf_model.best_estimator_,
        'imputer': imputer,
        'power_transformer': power_transformer,
        'selector': selector,
        'gender_encoder': gender_encoder,
        'gender_categories': gender_categories,
        'feature_names': X.columns.tolist(),
        'selected_features': X.columns[selector.get_support()].tolist()
    }, 'liver_model.pkl')

    # Evaluation
    y_pred = rf_model.predict(X_test_selected)
    y_prob = rf_model.predict_proba(X_test_selected)[:, 1]

    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


def model2():
    df = pd.read_csv(r'E:\Proj_4\venv\parkinsons - parkinsons.csv')
    X = df[['spread1', 'PPE', 'spread2', 'MDVP:Fo(Hz)']]
    y = df['status']

    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train with pipeline (auto-scaling)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    # Save the model to a file
    joblib.dump(model, 'parkinsons_model.pkl')
    print("Model saved as 'parkinsons_model.pkl'")

def model3():
    
    # Load and prepare data
    df = pd.read_csv(r'E:\Proj_4\venv\kidney_disease - kidney_disease.csv')

    # Verify target column
    print("Unique values in 'classification':", df['classification'].unique())

    # Convert target to binary - ensure correct mapping
    df['target'] = df['classification'].apply(lambda x: 1 if x == 'ckd' else 0)

    # Check class balance
    print("\nClass distribution:")
    print(df['target'].value_counts(normalize=True))

    # Remove problematic columns
    df = df.drop(['id', 'classification'], axis=1, errors='ignore')

    # Select features - based on medical relevance and your previous analysis
    selected_features = ['hemo', 'sg', 'al', 'sc']
    X = df[selected_features]
    y = df['target']

    # Verify no duplicate rows
    print("\nDuplicate rows:", df.duplicated().sum())
    df = df.drop_duplicates()

    # Preprocessing pipeline
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', make_pipeline(
                SimpleImputer(strategy='median'),
                StandardScaler()
            ), numeric_cols),
            ('cat', make_pipeline(
                SimpleImputer(strategy='most_frequent'),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            ), categorical_cols)
        ],
        remainder='drop'  # Explicitly drop unselected columns
    )

    # Model pipeline with regularization
    model = make_pipeline(
        preprocessor,
        LogisticRegression(
            penalty='l2',
            C=0.1,  # Stronger regularization
            max_iter=1000,
            random_state=42,
            solver='liblinear'
        )
    )

    # STRATIFIED K-Fold Cross Validation
    print("\nCross-validation results:")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Fold accuracies: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    # Final evaluation with holdout set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,  # Larger test set
        random_state=42,
        stratify=y
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print("\nTraining accuracy:", accuracy_score(y_train, train_pred))
    print("Test accuracy:", accuracy_score(y_test, test_pred))

    # Confusion matrix
    print("\nTest set confusion matrix:")
    print(confusion_matrix(y_test, test_pred))

    # Feature importance
    try:
        coefficients = model.named_steps['logisticregression'].coef_[0]
        feature_names = (numeric_cols +
                        list(model.named_steps['columntransformer']
                            .named_transformers_['cat']
                            .named_steps['onehotencoder']
                            .get_feature_names_out(categorical_cols)))

        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
        print("\nTop 10 most important features:")
        print(coef_df.head(10))
    except Exception as e:
        print("\nCould not extract coefficients:", e)

    # Save only the model to a pickle file
    print("\nSaving model to 'kidney_disease_model.pkl'...")
    with open('kidney_disease_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")
if __name__ == "__main__":
    train_and_save_model()
    model2()
    model3()
