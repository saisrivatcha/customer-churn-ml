import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/Telco-Customer-Churn.csv")

sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()
