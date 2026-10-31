import argparse
import matplotlib.pyplot as plt
from analysis._plot import load,save
p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--output',default='results/dask/memory.png');a=p.parse_args()
df=load(a.csv)
for (workload,workers),g in df.groupby(['workload','workers']):
    plt.plot(g.dataset,g.peak_rss_bytes/2**30,marker='o',label=f'{workload}, {workers} workers')
plt.xlabel('Input rows');plt.ylabel('Peak sampled process RSS (GiB)');plt.title('Dask memory footprint');plt.legend(fontsize=7);save(a.output)
