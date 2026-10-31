import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
p=Path(__file__).parent
x=pd.read_csv('/mnt/data/benchmark_evaluation/unified_summary.csv')
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7})
fig,axs=plt.subplots(1,3,figsize=(7.1,2.05),layout='constrained')
for ax,w in zip(axs,['aggregation','join','kmeans']):
 for f,worker,style in [('Dask',1,'-o'),('Spark',1,'-s'),('RAPIDS',1,'-^'),('Dask',4,'--o'),('Spark',4,'--s')]:
  d=x[(x.workload==w)&(x.framework==f)&(x.workers==worker)].sort_values('dataset')
  ax.plot(d.dataset/1e6,d.median_seconds,style,markersize=3,linewidth=1.2,label=f+(' (4 CPU workers)' if worker==4 else ' (1 worker)'))
 ax.set_title(w.capitalize());ax.set_xlabel('Rows (millions)');ax.grid(alpha=.25)
 ax.set_yscale('log' if w=='kmeans' else 'linear')
axs[0].set_ylabel('Median time (s)')
h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,bbox_to_anchor=(.5,-.12),frameon=False)
fig.savefig(p/'runtime.pdf',bbox_inches='tight')
plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(7.1,1.9),layout='constrained')
for ax,w in zip(axs,['aggregation','join','kmeans']):
 for f in ['Dask','Spark']:
  d=x[(x.workload==w)&(x.framework==f)&(x.dataset==2000000)].sort_values('workers')
  t1=float(d[d.workers==1].median_seconds.iloc[0]);ax.plot(d.workers,t1/d.median_seconds,'-o',markersize=3,label=f)
 ax.plot([1,8],[1,8],':',color='gray',label='Ideal')
 ax.set_title(w.capitalize());ax.set_xlabel('CPU workers');ax.set_xticks([1,2,4,8]);ax.grid(alpha=.25)
axs[0].set_ylabel('Speedup vs 1 worker')
h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,bbox_to_anchor=(.5,-.13),frameon=False)
fig.savefig(p/'scaling.pdf',bbox_inches='tight')
