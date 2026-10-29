import argparse
import json
from pathlib import Path
import numpy as np

PAIRS = ((0, 1), (0, 2), (1, 2))

def estimate(x, topk=3, min_abs_corr=0.15):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != 3:
        raise ValueError('x needs [N,3,D]')
    if x.shape[0] < 4:
        raise ValueError('traing at least 4')
    if not np.isfinite(x).all():
        raise ValueError('existing NaN/Inf')
    n, _, d = x.shape
    mean = x.mean(axis=0)
    std_raw = x.std(axis=0, ddof=1)
    std = np.maximum(std_raw, 1e-4)
    z = (x - mean) / std
    # Assign 0 to the constant dimension to prevent false numerical correlation
    z[:, std_raw < 1e-7] = 0
    edges = {}
    all_corr = {}
    for m, p in PAIRS:
        corr = (z[:, m].T @ z[:, p]) / (n-1)
        corr = np.clip(corr, -0.9999, 0.9999)
        all_corr[f'corr_{m}{p}'] = corr
        # The bidirectional Top-k takes the union, but each edge appears only once.
        mask = np.zeros_like(corr, dtype=bool)
        k = min(max(1, topk), d)
        rows = np.arange(d)[:, None]
        cols = np.argpartition(np.abs(corr), -k, axis=1)[:, -k:]
        mask[rows, cols] = True
        rows_by_col = np.argpartition(np.abs(corr), -k, axis=0)[-k:, :]
        mask[rows_by_col, np.arange(d)[None, :]] = True
        mask &= np.abs(corr) >= min_abs_corr
        for a, b in np.argwhere(mask):
            edges[(m, p, int(a), int(b))] = float(corr[a, b])
    if not edges:
        raise ValueError('There are no candidate matching edges; Please reduce--min-abs-corr')
    tuples = sorted(edges)
    rho = np.array([edges[t] for t in tuples], dtype=np.float64)
    mi = -0.5 * np.log1p(-np.minimum(rho**2, 0.9998))
    arrays = dict(
        mean=mean.astype('float32'), std=std.astype('float32'),
        edge_m=np.array([t[0] for t in tuples], dtype='int32'),
        edge_n=np.array([t[1] for t in tuples], dtype='int32'),
        edge_d=np.array([t[2] for t in tuples], dtype='int32'),
        edge_k=np.array([t[3] for t in tuples], dtype='int32'),
        edge_rho=rho.astype('float32'), edge_mi=mi.astype('float32'),
    )
    return arrays, all_corr


def diagnostic(x, out_prefix, all_corr, seed=42):
    try:
        from scipy.stats import skew, kurtosis, normaltest, spearmanr
        from matplotlib import pyplot as plt
    except ImportError:
        print('scipy/matplotlib is not installed. Skip Q-Q/ distribution graph (the statistical file is still saved).')
        return
    rng = np.random.default_rng(seed)
    n, _, d = x.shape
    summary = {}
    for m in range(3):
        v = x[:, m, :]
        valid = v.std(axis=0) > 1e-6
        q = np.nan_to_num(skew(v[:, valid], axis=0), nan=0)
        k = np.nan_to_num(kurtosis(v[:, valid], axis=0), nan=0)
        rejected = float(np.mean(normaltest(v[:, valid], axis=0).pvalue < 0.05)) if n>=20 and valid.any() else None
        summary[str(m)] = {'skew_median_abs': float(np.median(abs(q))) if len(q) else None,
                           'excess_kurtosis_median_abs':float(np.median(abs(k))) if len(k) else None,
                           'fraction_normaltest_reject_05':rejected,
                           'valid_dims':int(valid.sum())}
    Path(str(out_prefix)+'_distribution.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,corr in all_corr.items():
        fig,ax=plt.subplots(figsize=(5,4))
        ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
        ax.set(title=name,xlabel='other modality dimension',ylabel='source modality dimension')
        fig.tight_layout();fig.savefig(str(out_prefix)+'_'+name+'.png',dpi=130);plt.close(fig)
    # Display the normal Q-Q plot/histogram of a few dimensions. Marginal normality cannot be used to replace the joint Gaussian test.
    from scipy.stats import probplot
    fig,axs=plt.subplots(2,3,figsize=(11,6))
    for m in range(3):
        j=int(rng.integers(d)); val=x[:,m,j]
        axs[0,m].hist(val,bins=35);axs[0,m].set_title(f'modal {m} dim {j}')
        probplot(val,dist='norm',plot=axs[1,m])
    fig.tight_layout();fig.savefig(str(out_prefix)+'_marginals.png',dpi=130);plt.close(fig)


def load_unique(path):
    with np.load(path,allow_pickle=False) as f:
        x=np.asarray(f['x'])
        if 'drug_ids' in f:
            ids=f['drug_ids'].reshape(-1)
            if len(ids)!=len(x): raise ValueError('drug_ids长度与x不一致')
            _, unique_idx=np.unique(ids,return_index=True)
            x=x[np.sort(unique_idx)]
            print(f'After taking weight reduction drugs {len(x)},（input {len(ids)} ）')
        else:
            print('Warning: No drug_ids, assumed to have been de-duplicated; Repeated medication can cause statistical bias')
    return x


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--input',required=True)
    a.add_argument('--output',required=True)
    a.add_argument('--topk',type=int,default=3)
    a.add_argument('--min-abs-corr',type=float,default=0.15)
    a.add_argument('--no-plots',action='store_true')
    args=a.parse_args()
    x=load_unique(args.input)
    arrays, corr=estimate(x,args.topk,args.min_abs_corr)
    np.savez_compressed(args.output,**arrays)
    print('saved:',args.output,'samples=',len(x),'edges=',len(arrays['edge_m']))
    if not args.no_plots:
        diagnostic(x,Path(args.output).with_suffix(''),corr)

if __name__=='__main__':
    main()
