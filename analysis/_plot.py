import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load(path):
    df=pd.read_csv(path)
    return df.groupby(['framework','workload','dataset','workers'],as_index=False).agg(seconds=('seconds','median'),peak_rss_bytes=('peak_rss_bytes','median'))

def save(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    plt.tight_layout();plt.savefig(path,dpi=160);plt.close();print('Saved',path)
