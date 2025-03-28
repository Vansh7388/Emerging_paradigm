import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("final_data.csv")

# Restrict to Tech 1.0 (1995–2002) and Tech 2.0 (2003–2012)
df = df[(df['Year'] >= 1995) & (df['Year'] <= 2012) &
        (df['Country'].isin(['USA','China','Japan','Russia']))].copy()

# Rename columns to simpler names for the analysis.
df.rename(columns={
    'R&D_Expenditure_Percentage': 'RD',
    'Patent_Apps_Residents': 'Patents',
    'HighTech_Exports_Percentage': 'HighTechExports'
}, inplace=True)

# Scale Patents (convert count to thousands for interpretability)
df['Patents_thousands'] = df['Patents'] / 1000.0

# Fit the OLS regression with country and year fixed effects.
model = smf.ols('GDP_Growth ~ RD + Patents_thousands + HighTechExports + C(Country) + C(Year)', 
                data=df).fit()

# Print the regression summary (for text-based insight)
print(model.summary())

# Generate predictions for the sample data
df['predicted_GDP_Growth'] = model.predict(df)

# Plot Actual vs. Predicted GDP Growth
plt.figure(figsize=(8, 6))
plt.scatter(df['GDP_Growth'], df['predicted_GDP_Growth'], color='blue', alpha=0.7)
plt.plot([df['GDP_Growth'].min(), df['GDP_Growth'].max()], 
         [df['GDP_Growth'].min(), df['GDP_Growth'].max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual GDP Growth (%)")
plt.ylabel("Predicted GDP Growth (%)")
plt.title("Actual vs. Predicted GDP Growth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optionally, plot residuals to assess prediction errors.
plt.figure(figsize=(8, 6))
plt.scatter(df['predicted_GDP_Growth'], df['GDP_Growth'] - df['predicted_GDP_Growth'], 
            color='green', alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted GDP Growth (%)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.grid(True)
plt.tight_layout()
plt.show()
