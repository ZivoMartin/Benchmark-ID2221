"""Validate benchmark CSV for missing runs and inconsistent outputs."""
import argparse
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--repetitions',type=int,default=3);args=p.parse_args()
    df=pd.read_csv(args.csv)
    if df.empty:raise SystemExit('No results')
    assert (df.status=='ok').all()
    for key,group in df.groupby(['framework','workload','dataset','workers']):
        assert len(group)==args.repetitions,(key,'incorrect repetition count',len(group))
        assert group.run.nunique()==args.repetitions,(key,'duplicate runs')
        assert group.rows.nunique()==1,(key,'row count changed')
        assert group.checksum.max()-group.checksum.min() <= 1e-6*max(1,abs(group.checksum.mean())),(key,'checksum changed')
    print(f'PASS: {len(df)} measurements, {df.groupby(["framework","workload","dataset","workers"]).ngroups} configurations')
if __name__=='__main__':main()
