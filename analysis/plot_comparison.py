import argparse
import matplotlib.pyplot as plt
from analysis._plot import load,save
p=argparse.ArgumentParser();p.add_argument('csv',nargs='+');p.add_argument('--output',default='results/comparison.png');a=p.parse_args()
import pandas as pd
df=pd.concat([load(path) for path in a.csv],ignore_index=True)
for (framework,workload,workers),g in df.groupby(['framework','workload','workers']):
    plt.plot(g.dataset,g.seconds,marker='o',label=f'{framework} {workload} {workers}w')
plt.xlabel('Input rows');plt.ylabel('Median runtime (s)');plt.title('Framework comparison');plt.legend(fontsize=7);save(a.output)
