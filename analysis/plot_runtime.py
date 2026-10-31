import argparse
import matplotlib.pyplot as plt
from analysis._plot import load,save
p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--output',default='results/dask/runtime.png');a=p.parse_args()
df=load(a.csv)
for (workload,workers),g in df.groupby(['workload','workers']):
    plt.plot(g.dataset,g.seconds,marker='o',label=f'{workload}, {workers} workers')
plt.xlabel('Input rows');plt.ylabel('Median runtime (s)');plt.title('Dask runtime scaling');plt.legend(fontsize=7);save(a.output)
