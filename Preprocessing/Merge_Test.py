#%%
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

bg_files = [
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_bg6.csv',

]

sig_files = [
    r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_Added_Data\processed_filtered_sig6.csv',

]



#create different labels for sig and bg
bg_data = pd.concat([pd.read_csv(f).assign(label=0) for f in bg_files], ignore_index=True)
sig_data = pd.concat([pd.read_csv(f).assign(label=1) for f in sig_files], ignore_index=True)



data = pd.concat([bg_data, sig_data], ignore_index=True)
print(sig_data.shape)
print(bg_data.shape)



output_path = r"D:\Year3\BSc Project\Particle-Machine-Learning\nonbalanced_test_added.csv"
data.to_csv(output_path, index=False)

print(f"output successfully {output_path}")

# %%
