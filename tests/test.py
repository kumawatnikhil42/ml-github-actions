import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def test_data_loading():
    data = pd.read_csv("insurance.csv")
    assert not data.empty, "Dataset is empty"
    assert "charges" in data.columns, "'charges' column is missing from the dataset"
    assert len(data) > 0, "Dataset has no rows"

def test_data_preprocessing():
    data = pd.read_csv("insurance.csv")
    target = 'charges' 
    X = data.drop(columns=[target])
    y = data[target]
    
    categorical_features = ['sex', 'smoker', 'region']
    for feature in categorical_features:
        X[feature] = LabelEncoder().fit_transform(X[feature])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    assert X.shape[0] == len(y), "Mismatch in number of rows between features and target"
    assert X.select_dtypes(include=['float64', 'int64']).shape[1] == len(X.columns), "Non-numeric values present in the feature set"

def test_model_training():
    data = pd.read_csv("insurance.csv")
    target = 'charges' 
    X = data.drop(columns=[target])
    y = data[target]
    
    categorical_features = ['sex', 'smoker', 'region']
    for feature in categorical_features:
        X[feature] = LabelEncoder().fit_transform(X[feature])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    assert hasattr(model, 'coef_'), "Model failed to train"


def test_model_performance():
    data = pd.read_csv("insurance.csv")
    target = 'charges' 
    X = data.drop(columns=[target])
    y = data[target]
    
    categorical_features = ['sex', 'smoker', 'region']
    for feature in categorical_features:
        X[feature] = LabelEncoder().fit_transform(X[feature])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    assert mae < 5000, f"MAE is too high: {mae}"
    assert mse < 35000000, f"MSE is too high: {mse}"
    assert r2 > 0.7, f"R² score is too low: {r2}"

