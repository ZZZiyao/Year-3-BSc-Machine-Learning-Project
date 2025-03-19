import pandas as pd
import os
import matplotlib.pyplot as plt

bg1 = pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\filtered_bg1.csv')
sig1=pd.read_csv(r'D:\Year3\BSc Project\Particle-Machine-Learning\filtered_sig1.csv')


output_dir = r"D:\Year3\BSc Project\Particle-Machine-Learning\Kaon"
os.makedirs(output_dir, exist_ok=True)


variables = [
    ('kaon_PX', 'Kaon PX (MeV/c)', 'Kaon PX Distribution'),
    ('kaon_PY', 'Kaon PY (MeV/c)', 'Kaon PY Distribution'),
    ('kaon_PZ', 'Kaon PZ (MeV/c)', 'Kaon PZ Distribution'),
    ('kaon_P', 'Kaon P (MeV/c)', 'Kaon Total Momentum Distribution'),
    ('kaon_PT', 'Kaon Transverse Momentum (MeV/c)', 'Kaon Transverse Momentum Distribution'),
    ('kaon_PE', 'Kaon E (MeV)', 'Kaon Total Energy Distribution'),
    ('kaon_M', 'Kaon M (MeV/c^2)', 'Kaon Mass Distribution'),
    ('kaon_IPCHI2_OWNPV', 'Kaon IP Chi2 OWN PV', 'Kaon Impact Parameter Chi2 Distribution'),

]


for var, xlabel, title in variables:
    plt.hist(bg1[var], bins=50, alpha=0.7, label='background', color='blue',density=True)
    plt.hist(sig1[var], bins=50, alpha=0.9, label='signal', color='green',density=True)
    plt.xlabel(xlabel)
    plt.ylabel('Counts')
    plt.title(title)
    plt.legend()


    output_file = os.path.join(output_dir, f'{var}_Distribution.png')
    plt.savefig(output_file)
    plt.close()

    print(f"Plot saved to: {output_file}")
