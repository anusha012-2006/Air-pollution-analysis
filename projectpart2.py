import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# ---------------- LOAD DATA ---------------- #

df = pd.read_csv("airpollution01.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("Columns:", df.columns)
print(df.head())

# ---------------- DATA CLEANING ---------------- #

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert date column
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

print("After Cleaning:")
print(df.info())

# ---------------- VISUALIZATION ---------------- #

# Histogram
df.hist(figsize=(10,8))
plt.suptitle("Pollution Data Distribution")
plt.show()

# Heatmap
numeric_df = df.select_dtypes(include='number')

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("Numeric Columns:", numeric_df.columns)

# Time trend #Shows how pollution changes day-by-day / month-by-month
if 'date' in df.columns:
    plt.figure()
    for col in numeric_df.columns:
        plt.plot(df['date'], df[col], label=col)
    plt.legend()
    plt.title("Pollution Trend Over Time")
    plt.show()

# ---------------- EDA ---------------- #

print(df.describe())
print("Skewness:\n", numeric_df.skew())

print("\n" + "="*50)
print("EDA RESULTS")
print("="*50)

pm_col = None
for col in df.columns:
    if 'pm2.5' in col.lower():
        pm_col = col
        break

print("Detected PM Column:", pm_col)

#AIR QUALITY CATEGORY
def air_quality(pm):
    if pm <= 50:
        return "Good"
    elif pm <= 100:
        return "Moderate"
    else:
        return "Poor"

if pm_col:
    df['Air_Quality'] = df[pm_col].apply(air_quality)
    print(df[[pm_col, 'Air_Quality']].head())

    sns.countplot(x='Air_Quality', data=df)
    plt.title("Air Quality Categories")
    plt.show()
else:
    print("PM2.5 column not found!")
    
#statics
print("Mean:\n", numeric_df.mean())
print("Median:\n", numeric_df.median())
print("Standard Deviation:\n", numeric_df.std())

# ---------------- OUTLIERS ---------------- #

plt.figure()
sns.boxplot(data=numeric_df)
plt.title("Outlier Detection")
plt.show()

# ---------------- MACHINE LEARNING ---------------- #

# Use only numeric data
df_ml = numeric_df.copy()

# Target column
target = pm_col if pm_col else df_ml.columns[-1]

X = df_ml.drop(columns=[target])
y = df_ml[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Model Accuracy:", model.score(X_test, y_test))
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
#visulazation
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()


# ---------------- EXTRA (UNIQUE FEATURES) ---------------- #

# Feature importance
importance = pd.Series(model.coef_, index=X.columns)
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()

# Highest pollution record
if pm_col:
    max_row = df.loc[df[pm_col].idxmax()]
    print("Highest Pollution Record:\n", max_row)

# Prediction demo
sample = X.iloc[0:1]
pred = model.predict(sample)
print("Actual:", y.iloc[0])
print("Predicted:", pred[0])

