import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = pd.read_csv("insurance.csv") 

target = 'charges' 
x = data.drop(columns=[target])
y = data[target]


categorical_features = ['sex', 'smoker', 'region'] 
numerical_features = ['age', 'bmi', 'children']  

lb=LabelEncoder()

for feature in categorical_features:
    x[feature] = LabelEncoder().fit_transform(x[feature])

scaler = StandardScaler()
sc_x=scaler.fit_transform(x)
x_sc=pd.DataFrame(sc_x,columns=x.columns)

x_train, x_test, y_train, y_test = train_test_split(x_sc, y, test_size=0.2, random_state=42)


model_type = 'linear'  

if model_type == 'linear':
    model = LinearRegression()
elif model_type == 'random_forest':
    model = RandomForestRegressor(random_state=42)


model.fit(x_train, y_train)


y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")