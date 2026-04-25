import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load Dataset
url = 'https://github.com/ybifoundation/Dataset/raw/main/House%20Prices.csv'
df = pd.read_csv(url)

# 1. CLEANING: Remove rows with missing prices
df = df.dropna(subset=['Price'])

# 2. SELECTION: Define your features
# Choose features that actually impact price
features = ['Crime_Rate', 'Resid_Area', 'Air_Qual', 'Room_Num', 'Age', 'Teachers', 'Poor_Prop']
X = df[features]
y = df['Price']

# 3. TRAINING: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODELING: Fit the Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 5. PREDICTION
y_pred = model.predict(X_test)
plt.figure(figsize=(10,6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Model Accuracy: Actual vs. Predicted House Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()