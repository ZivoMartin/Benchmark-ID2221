import argparse
import matplotlib.pyplot as plt
from analysis._plot import load,save
p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--output',default='results/dask/scaling.png');a=p.parse_args()
df=load(a.csv)
for (workload,size),g in df.groupby(['workload','dataset']):
    baseline=g[g.workers==1]
    if baseline.empty:continue
    plt.plot(g.workers,float(baseline.seconds.iloc[0])/g.seconds,marker='o',label=f'{workload}, {size} rows')
plt.xlabel('Workers');plt.ylabel('Speedup relative to 1 worker');plt.title('Dask local scaling');plt.legend(fontsize=7);save(a.output)
