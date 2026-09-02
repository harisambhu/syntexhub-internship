# ==========================================
# HOUSE PRICE PREDICTION USING MACHINE LEARNING
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("Housing.csv")

print("HOUSE PRICE PREDICTION PROJECT")
print("=" * 50)

print("\nDataset Shape:")
print(data.shape)


# ==========================================
# 2. CHECK DATA
# ==========================================

print("\nFirst 5 Rows:")
print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())


# ==========================================
# 3. CONVERT CATEGORICAL DATA
# ==========================================

categorical_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

for column in categorical_columns:
    data[column] = data[column].map({"yes": 1, "no": 0})


# Convert furnishing status into numerical values
data["furnishingstatus"] = data["furnishingstatus"].map({
    "unfurnished": 0,
    "semi-furnished": 1,
    "furnished": 2
})


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

X = data.drop("price", axis=1)
y = data["price"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("price")


# ==========================================
# 5. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==========================================
# 6. TRAIN MACHINE LEARNING MODEL
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")


# ==========================================
# 7. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 8. EVALUATE MODEL
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("=" * 50)

print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R² Score:", round(r2, 4))


# ==========================================
# 9. SAMPLE HOUSE PRICE PREDICTION
# ==========================================

sample_house = pd.DataFrame([{
    "area": 5000,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "mainroad": 1,
    "guestroom": 0,
    "basement": 0,
    "hotwaterheating": 0,
    "airconditioning": 1,
    "parking": 2,
    "prefarea": 1,
    "furnishingstatus": 1
}])

predicted_price = model.predict(sample_house)

print("\nSAMPLE HOUSE")
print("=" * 50)

print("Area: 5000 sq ft")
print("Bedrooms: 3")
print("Bathrooms: 2")
print("Stories: 2")
print("Parking: 2")
print("Air Conditioning: Yes")
print("Furnishing Status: Semi-Furnished")

print("\nPredicted House Price:")
print(round(predicted_price[0], 2))