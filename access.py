import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
df = pd.read_csv(r"C:\Users\91944\Desktop\DA Project-1\shopping_trends.csv")
"""
print(df.head())
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Check for duplicates
print(df.duplicated().sum())

df = df.drop_duplicates()

#print mean,median,avg...
print(df.describe())

print(df['Gender'].value_counts())
print(df['Location'].value_counts())

#histogram
df['Age'].plot(kind='hist', bins=20, title='Age Distribution')
plt.show()

#boxplot
df.boxplot(column='Purchase Amount (USD)')
plt.show()

#bar graph
df['Gender'].value_counts().plot(kind='bar', title='Gender Distribution')
plt.show()
"""
"""
import seaborn as sns
correlation = df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
"""
"""
# Create age groups
bins = [18, 30, 40, 50, 60, 70]
labels = ['18-30', '31-40', '41-50', '51-60', '61-70']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Analyze purchase behavior by age group
age_purchase = df.groupby('Age Group')['Purchase Amount (USD)'].mean()
print(age_purchase)
"""

#Gender and Purchase Amount
#gender_purchase = df.groupby('Gender')['Purchase Amount (USD)'].mean()
#print(gender_purchase)

#most purchased items
#popular_items = df['Item Purchased'].value_counts().head(10)
#print(popular_items)

#subscription status influence on amount spent.
#subscription_purchase = df.groupby('Subscription Status')['Purchase Amount (USD)'].mean()
#print(subscription_purchase)

"""
#impact of discounts and promo codes on purchase amounts.
discount_purchase = df.groupby('Discount Applied')['Purchase Amount (USD)'].mean()
print(discount_purchase)

promo_purchase = df.groupby('Promo Code Used')['Purchase Amount (USD)'].mean()
print(promo_purchase)
"""
"""
#segment customers based on their purchasing behavior (such as frequency of purchases and amount spent)

kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(df[['Purchase Amount (USD)', 'Previous Purchases']])
print(df[['Customer ID', 'Cluster']].head(20))
"""
"""
from sklearn.linear_model import LinearRegression
X = df[['Age', 'Previous Purchases', 'Review Rating']]  # Features
y = df['Purchase Amount (USD)']  # Target

model = LinearRegression()
model.fit(X, y)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
"""

#Top 5 Most Purchased Items
#top_items = df['Item Purchased'].value_counts().head(5)
#print(top_items)

#Most Popular Category Based on Revenue
#popular_categories = df.groupby('Category')['Purchase Amount (USD)'].sum().sort_values(ascending=False)
#print(popular_categories)

#seasonal influence on revenue
#seasonal_trends = df['Season'].value_counts()
#print(seasonal_trends)

"""
# Compare average purchase amount with and without discounts
avg_purchase_discount = df.groupby('Discount Applied')['Purchase Amount (USD)'].mean()
print("Average Purchase Amount With and Without Discounts:\n", avg_purchase_discount)

# Compare average purchase amount with and without promo codes
avg_purchase_promo = df.groupby('Promo Code Used')['Purchase Amount (USD)'].mean()
print("Average Purchase Amount With and Without Promo Codes:\n", avg_purchase_promo)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Discount Effectiveness Plot
sns.barplot(x=avg_purchase_discount.index, y=avg_purchase_discount.values, ax=axes[0], palette="viridis")
axes[0].set_title("Avg Purchase Amount With and Without Discounts")
axes[0].set_xticklabels(["No Discount", "Discount Applied"])
axes[0].set_ylabel("Avg Purchase Amount (USD)")

# Promo Code Effectiveness Plot
sns.barplot(x=avg_purchase_promo.index, y=avg_purchase_promo.values, ax=axes[1], palette="coolwarm")
axes[1].set_title("Avg Purchase Amount With and Without Promo Codes")
axes[1].set_xticklabels(["No Promo", "Promo Used"])
axes[1].set_ylabel("Avg Purchase Amount (USD)")

plt.tight_layout()
plt.show()
"""

"""
#A bar chart will display the most popular items in your dataset.
top_items = df['Item Purchased'].value_counts().head(5)

# Plot a bar chart
plt.figure(figsize=(10, 6))
top_items.plot(kind='bar', color='skyblue')
plt.title('Top 5 Most Purchased Items')
plt.xlabel('Item')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=45)
plt.show()
"""


# Plot the average purchase amount for each season
"""
avg_purchase_season = df.groupby('Season')['Purchase Amount (USD)'].mean()

plt.figure(figsize=(10, 6))
avg_purchase_season.plot(kind='bar', color='lightcoral')
plt.title('Average Purchase Amount by Season')
plt.xlabel('Season')
plt.ylabel('Average Purchase Amount (USD)')
plt.xticks(rotation=45)
plt.show()

"""

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot a heatmap of correlations
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

















