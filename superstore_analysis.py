import pandas as pd
df = pd.read_csv("C:/Users/dariu/desktop/Proiect 1/Superstore.csv")
print(df.head())
print(df.columns)
print(df.info())
print(df.describe())

print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique())
print(df['Category'].value_counts())

# Conversie date și coloane calendaristice
if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
if "Ship Date" in df.columns and "Order Date" in df.columns:
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df['Shipping Duration'] = (df['Ship Date'] - df['Order Date']).dt.days
    print(df['Shipping Duration'].describe())

#  # We do not ensure that Sales and Profit are numeric
for col in ["Sales", "Profit"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
# 2) Curățare (asigură tipuri corecte)
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")
#Ce face codul:
#Grupează datele pe Region și State.
#Calculează totalul de Sales și Profit.
#Creează grafice cu bare pentru o prezentare clară.

# 3) Agregare pe Regiuni

if "Region" in df.columns:
    region_sales = df.groupby("Region").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum")
    ).sort_values("Sales", ascending=False)

    print("Vânzări și profit pe Regiuni:")
    print(region_sales)

    # Vizualizare
    import matplotlib.pyplot as plt
    import seaborn as sns
    region_sales[["Sales", "Profit"]].plot(kind="bar", figsize=(8,5), title="Sales & Profit by Region")
    plt.ylabel("USD")
    plt.show()

# 4) Agregare pe State (dacă există coloana)
if "State" in df.columns:
    state_sales = df.groupby("State").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum")
    ).sort_values("Sales", ascending=False).head(10)  # top 10 state după vânzări

    print("\nTop 10 State după vânzări:")
    print(state_sales)

    # Vizualizare
    state_sales["Sales"].plot(kind="bar", figsize=(10,5), title="Top 10 State by Sales")
    plt.ylabel("USD")
    plt.show()
#  Profit_Margin on each row (avoid division by 0)
import numpy as np
df["Profit_Margin"] = np.where(df["Sales"].fillna(0) != 0,
                               df["Profit"] / df["Sales"],
                               np.nan)


# KPI global (pe tot dataset-ul)
total_sales = df["Sales"].sum(skipna=True)
total_profit = df["Profit"].sum(skipna=True)
kpi_profit_margin = (total_profit / total_sales) if total_sales else np.nan

print(f"Total Sales:  {total_sales:,.2f}")
print(f"Total Profit: {total_profit:,.2f}")
print("Profit Margin (global): " +
      (f"{kpi_profit_margin:.2%}" if pd.notna(kpi_profit_margin) else "N/A"))

# Agregări pe categorie / sub-categorie (dacă există coloanele)
if "Category" in df.columns:
    by_cat = (df.groupby("Category", dropna=False)
                .agg(Sales=("Sales", "sum"),
                     Profit=("Profit", "sum"))
                .assign(Profit_Margin=lambda x: x["Profit"] / x["Sales"]))
    print("\nPe Category:")
    print(by_cat.sort_values("Sales", ascending=False))

# Agregări lunare (dacă ai creat Month)
if "Month" in df.columns:
    by_month = (df.groupby("Month")
                  .agg(Sales=("Sales","sum"),
                       Profit=("Profit","sum"))
                  .assign(Profit_Margin=lambda x: x["Profit"] / x["Sales"]))
    print("\nPe lună (ultimele 6):")
    print(by_month.tail(6))

df["Profit_Margin"] = np.where(df["Sales"].fillna(0) != 0,
                               df["Profit"] / df["Sales"],
                               np.nan)
print(df[["Sales", "Profit", "Profit_Margin"]].describe())

for col in ["Sales", "Profit", "Discount"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

import matplotlib.pyplot as plt
import numpy as np
# Histogramă: Distribuția vânzărilor
plt.figure(figsize=(7,5))
plt.hist(df["Sales"].dropna(), bins=30, color="skyblue", edgecolor="black")
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter plot: Discount vs Profit
plt.figure(figsize=(7,5))
plt.scatter(df["Discount"], df["Profit"], alpha=0.3, c="blue")
plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.axhline(0, color="red", linestyle="--", linewidth=1)   # linia zero profit
plt.tight_layout()
plt.show()

#  Matrice de corelație pentru coloanele numerice
num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
corr = num_df.corr(numeric_only=True)

plt.figure(figsize=(8,6))
im = plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix (numeric)")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Category')
plt.title('Count of Each Category')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Category', y='Sales')
plt.title('Sales Distribution by Category')
plt.show()

import numpy as np
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Profit', y='Sales', hue='Category', size='Quantity', sizes=(20, 200), alpha=0.6)
plt.title('Profit vs Sales')
plt.show()

import plotly.express as px
fig = px.pie(df, names='Category', title='Category Distribution')
fig.show()

import numpy as np
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Month'] = df['Order Date'].dt.month
monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
plt.figure(figsize=(10,6))
sns.lineplot(data=monthly_sales, x='Month', y='Sales', marker='o')
plt.title('Monthly Sales Trend')
plt.show()



import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("C:/Users/dariu/desktop/Proiect 1/Superstore.csv")
df = df.dropna()
df = df.select_dtypes(include=[np.number])
X = df.drop('Sales', axis=1)
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
correlation_matrix = df.corr()
print(correlation_matrix)

import plotly.express as px
fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title='Correlation Matrix')
fig.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Users/dariu/desktop/Proiect 1/Superstore.csv")
plt.figure(figsize=(10,6))
sns.histplot(df['Sales'], bins=30, kde=True)
plt.title('Sales Distribution')
plt.show()



