import matplotlib.pyplot as plt
import collections
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

bg1 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg1.csv')
# bg2 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg2.csv')
# bg3 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg3.csv')
# bg4= pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg4.csv')
# bg5= pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg5.csv')
# bg6= pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_bg6.csv')

sig1=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig1.csv')
# sig2=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig2.csv')
# sig3=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig3.csv')
# sig4=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig4.csv')
# sig5=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig5.csv')
# sig6=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\Filtered_data\filtered_sig6.csv')

particle_counts = collections.Counter(sig1.kaon_MC_GD_MOTHER_ID)

particles = list(particle_counts.keys())
counts = list(particle_counts.values())


plt.figure(figsize=(15, 6))
plt.bar(range(len(particles)), counts, width=0.8, align='center')  
plt.xlabel("Particle ID")
plt.ylabel("Count")
plt.title("Frequency of kaon_MC_GD_GD_MOTHER_ID")
plt.xticks(range(len(particles)), particles, rotation=90)  
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()






