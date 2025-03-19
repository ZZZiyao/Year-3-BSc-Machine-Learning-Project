import pandas as pd

bg_files = [
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_bg6.csv',

]

sig_files = [
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_sig6.csv',

]


bg_data = pd.concat([pd.read_csv(f).assign(label=0) for f in bg_files], ignore_index=True)
sig_data = pd.concat([pd.read_csv(f).assign(label=1) for f in sig_files], ignore_index=True)

min_samples = min(len(bg_data), len(sig_data))


bg_data_sampled = bg_data.sample(n=min_samples, random_state=42)
sig_data_sampled = sig_data.sample(n=min_samples, random_state=42)  


balanced_data = pd.concat([bg_data_sampled, sig_data_sampled], ignore_index=True)

output_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\balanced_test_added.csv"
balanced_data.to_csv(output_path, index=False)

print(f"Balanced dataset saved successfully: {output_path}")
print(balanced_data['label'].value_counts())  
