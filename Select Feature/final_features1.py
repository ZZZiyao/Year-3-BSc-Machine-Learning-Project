import pandas as pd

importance_df = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\See Importance\permutation_top100_importance.csv')
import matplotlib.pyplot as plt

importance_df = importance_df.head(20)
importance_df = importance_df.sort_values(by='Importance', ascending=True)


plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.tight_layout()  

plt.xlabel('Permutation Importance')
plt.ylabel('Feature')
plt.title('Feature Importance based on Permutation Importance')
plt.grid(axis='x', linestyle='--', alpha=0.7)

for index, value in enumerate(importance_df['Importance']):
    plt.text(value, index, f'{value:.3f}', va='center')

plt.show()