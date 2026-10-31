from pathlib import Path
import pandas as pd,numpy as np,matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

R=Path(__file__).parent
files={'Dask':'/mnt/data/measurements(3).csv','Spark':'/mnt/data/measurements(5).csv','RAPIDS':'/mnt/data/measurements(4).csv'}
dfs=[]

for name,path in files.items():
 d=pd.read_csv(path);d['framework']=name;dfs.append(d)

df=pd.concat(dfs,ignore_index=True)
df.to_csv(R/'all_measurements.csv',index=False)
group=['framework','workload','dataset','workers','partitions']
s=df.groupby(group,as_index=False).agg(runs=('seconds','count'),median_seconds=('seconds','median'),mean_seconds=('seconds','mean'),std_seconds=('seconds','std'),min_seconds=('seconds','min'),max_seconds=('seconds','max'),median_peak_rss_bytes=('peak_rss_bytes','median'),median_delta_rss_bytes=('delta_rss_bytes','median'),rows=('rows','first'),checksum=('checksum','median'))
s.to_csv(R/'unified_summary.csv',index=False)
colorset={'Dask':'#3273a8','Spark':'#de7132','RAPIDS':'#329b79'}

for work in ['aggregation','join','kmeans']:
 fig,ax=plt.subplots(figsize=(7.4,4.3))
 for fw in files:
  a=s[(s.workload==work)&(s.framework==fw)&(s.workers==1)].sort_values('dataset');ax.plot(a.dataset/1e6,a.median_seconds,marker='o',label=fw,color=colorset[fw],linewidth=2)
 ax.set(xlabel='Dataset size (million rows)',ylabel='Median runtime (seconds)',title=f'{work.title()}: single-worker CPU vs single-GPU RAPIDS');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(R/f'runtime_{work}.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7.4,4.3))
 for fw in ['Dask','Spark']:
  for workers in [1,2,4,8]:
   a=s[(s.workload==work)&(s.framework==fw)&(s.workers==workers)].sort_values('dataset');ax.plot(a.dataset/1e6,a.median_seconds,marker='o',label=f'{fw} {workers} workers',color=colorset[fw],alpha=.35+.08*workers,linestyle={1:'-',2:'--',4:':',8:'-.'}[workers])
 ax.set(xlabel='Dataset size (million rows)',ylabel='Median runtime (seconds)',title=f'{work.title()}: CPU worker configurations');ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(R/f'workers_{work}.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7.4,4.3))
 for fw in ['Dask','Spark']:
  base=s[(s.workload==work)&(s.framework==fw)&(s.workers==1)].set_index('dataset').median_seconds
  for workers in [2,4,8]:
   a=s[(s.workload==work)&(s.framework==fw)&(s.workers==workers)].sort_values('dataset');ax.plot(a.dataset/1e6,[base.loc[x]/y for x,y in zip(a.dataset,a.median_seconds)],marker='o',label=f'{fw} {workers} vs 1',color=colorset[fw],linestyle={2:'--',4:':',8:'-.'}[workers])
 ax.axhline(1,color='gray',linewidth=1);ax.set(xlabel='Dataset size (million rows)',ylabel='CPU speedup relative to 1 worker',title=f'{work.title()}: CPU parallel scaling');ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(R/f'scaling_{work}.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7.4,4.3))
 for fw in files:
  a=s[(s.workload==work)&(s.framework==fw)&(s.workers==1)].sort_values('dataset');ax.plot(a.dataset/1e6,a.median_peak_rss_bytes/2**20,marker='o',label=fw,color=colorset[fw])
 ax.set(xlabel='Dataset size (million rows)',ylabel='Median sampled process RSS (MiB)',title=f'{work.title()}: host memory (not GPU VRAM)');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(R/f'host_memory_{work}.png',dpi=180);plt.close(fig)
# GPU memory separate
fig,ax=plt.subplots(figsize=(7.4,4.3))
for work in ['aggregation','join','kmeans']:
 a=df[(df.framework=='RAPIDS')&(df.workload==work)].groupby('dataset').peak_gpu_bytes.median();ax.plot(a.index/1e6,a.values/2**20,marker='o',label=work)
ax.set(xlabel='Dataset size (million rows)',ylabel='Median sampled GPU memory (MiB)',title='RAPIDS device memory, including persistent allocations');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(R/'gpu_memory.png',dpi=180);plt.close(fig)
# audits
issues=[]
for fw,d in zip(files,dfs):
 expected=144 if fw!='RAPIDS' else 36
 if len(d)!=expected:issues.append(f'{fw}: expected {expected} runs, got {len(d)}')
 if not (d.status=='ok').all():issues.append(f'{fw}: failed statuses')
 if d.duplicated(['workload','dataset','workers','partitions','run']).any():issues.append(f'{fw}: duplicate run keys')
for (w,n),d in df.groupby(['workload','dataset']):
 if d.rows.nunique()!=1:issues.append(f'{w}/{n}: row mismatch')
 vals=d.groupby('framework').checksum.median()
 if not np.allclose(vals.values,vals.iloc[0],rtol=1e-8,atol=1e-6):issues.append(f'{w}/{n}: checksum mismatch')
(R/'audit.txt').write_text('Runs: Dask 144, Spark 144, RAPIDS 36. All statuses OK.\n'+'\n'.join(issues or ['No duplicate run keys, row-count discrepancies or checksum discrepancies detected.'])+'\nCaveat: checksum equivalence does not prove complete intermediate operation execution.\n')
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='ReportTitle',parent=styles['Title'],fontSize=19,leading=23,spaceAfter=15));styles.add(ParagraphStyle(name='SmallReport',parent=styles['BodyText'],fontSize=9,leading=13,spaceAfter=8));styles.add(ParagraphStyle(name='SectionReport',parent=styles['Heading2'],spaceBefore=15,spaceAfter=8))
story=[]
def p(t):story.append(Paragraph(t,styles['SmallReport']))
def h(t):story.append(Paragraph(t,styles['SectionReport']))
def fig(name,caption):
 story.append(Image(str(R/name),width=6.6*inch,height=3.84*inch));p(caption)
story.append(Paragraph('Benchmark evaluation: Dask, Spark and RAPIDS',styles['ReportTitle']))
p('NYC yellow taxi data | 100K, 500K, 1M and 2M rows | Aggregation, join and 10-iteration K-means | October 2026')
h('Executive findings')
p('All 324 measured executions succeeded: 144 Dask, 144 Spark and 36 RAPIDS. Every dataset/workload combination agrees on the reported row counts and numerical checksums across frameworks. This is a numerical consistency check, not proof that each engine executed identical intermediate operations.')
p('At 2M rows, single-worker K-means medians are 10.403 s (Dask), 145.320 s (Spark) and 3.060 s (RAPIDS). RAPIDS is approximately 3.40x faster than single-worker Dask and 47.5x faster than single-worker Spark in this configuration. The CPU and GPU machines are different, so these are observed end-to-end ratios, not isolated GPU acceleration factors.')
h('Experimental design and limitations')
p('CPU: 1, 2, 4 and 8 worker configurations, 16 partitions, one warmup and three measured runs per configuration; Python 3.14.8. RAPIDS: one Tesla T4 (15 GiB), one GPU worker, 16 partitions, three measured runs per dataset/workload after warmup; Python 3.13.16, cuDF 26.06.00. Different hardware and software versions confound direct architectural comparisons. No GPU worker-scaling experiment was conducted.')
p('Each median is based on only three repetitions. Timing variation should be treated descriptively; formal confidence intervals or statistically significant rankings are not justified. The available files do not provide complete CPU hardware specifications, energy usage, PCIe transfer timings, GPU utilization traces, or query execution plans.')
h('Cross-framework runtime')
for work in ['aggregation','join','kmeans']:
 fig(f'runtime_{work}.png',f'Figure: median end-to-end {work} runtime. CPU series show one-worker configurations; RAPIDS uses a single T4 GPU.')
h('2M-row runtime table')
rows=[['Workload','Dask 1w','Spark 1w','RAPIDS','Dask / GPU','Spark / GPU']]
for work in ['aggregation','join','kmeans']:
 vals={fw:float(s[(s.framework==fw)&(s.workload==work)&(s.dataset==2000000)&(s.workers==1)].median_seconds.iloc[0]) for fw in files}
 rows.append([work,*[f'{vals[x]:.3f} s' for x in ['Dask','Spark','RAPIDS']],f'{vals["Dask"]/vals["RAPIDS"]:.2f}x',f'{vals["Spark"]/vals["RAPIDS"]:.2f}x'])
t=Table(rows,colWidths=[1.15*inch,1*inch,1*inch,1*inch,1.05*inch,1.05*inch]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8edf3')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#b6c2cf')),('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]));story.append(t)
h('CPU worker scaling')
for work in ['aggregation','join','kmeans']:
 fig(f'scaling_{work}.png',f'Figure: CPU speedup of 2/4/8 worker configurations relative to the same framework with one worker. Speedup below 1 indicates regression.')
 fig(f'workers_{work}.png',f'Figure: absolute CPU runtime for all four worker configurations, {work}.')
h('Memory')
for work in ['aggregation','join','kmeans']:
 fig(f'host_memory_{work}.png',f'Figure: median sampled process RSS for the 1-worker CPU and RAPIDS processes. Host RSS is not a measure of GPU device memory; process-level sampling can miss short peaks and other processes.')
fig('gpu_memory.png','Figure: sampled RAPIDS GPU device memory, shown separately from host RSS. Memory pooling, baseline drift and sample granularity limit attribution to individual workloads.')
h('Interpretation')
p('Aggregation: RAPIDS runtime stays near 1.17–1.20 s over a 20x input-size range, consistent with fixed costs dominating these data sizes. This is a hypothesis, not a profiler-verified decomposition. Single-worker Spark increases more noticeably with dataset size; Dask remains relatively flat.')
p('Join: RAPIDS stays near 2.25–2.40 s across sizes. Spark is faster than RAPIDS at 100K, while the 2M single-worker Spark runtime exceeds RAPIDS. However, the join checksum is based on fare totals; this cannot independently establish that all intended join and shuffle work was materialized, especially if optimizer pruning is possible. Inspect query plans or materialize joined attributes before claiming shuffle performance.')
p('K-means: RAPIDS scales from about 1.32 s to 3.06 s as input grows 20x. It outperforms both single-worker CPU configurations at all four sizes. The iterative numerical workload is a plausible fit for GPU execution; exact causes require profiling. Report the different CPU/GPU hardware as a central threat to validity.')
h('Validity and reproducibility')
p('The row-count and checksum agreement across all frameworks is reassuring, but stronger validation would compare full aggregation outputs, joined columns, and K-means centroids or objective values. To reproduce the study, preserve raw data checksums, preparation seed and ordering, exact source revision, hardware specification, package freeze, partition counts, warmup logic and measurement boundaries. The RAPIDS config.txt was missing; the measurement records and gpu_info.txt recover some, but not all, environment details.')
p('Results here are derived from the uploaded run-level CSV files. The plots and unified summary are regenerated by build.py. All comparisons use observed medians of three runs. No claims of statistical significance or causal GPU advantage are made.')
doc=SimpleDocTemplate(str(R/'evaluation_report.pdf'),pagesize=(595,842),rightMargin=48,leftMargin=48,topMargin=50,bottomMargin=45);doc.build(story)
print('created',len(list(R.iterdir())),'files')
