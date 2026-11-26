"""Apply the independently frozen census bin convention to the 512 cold fork origins."""
import numpy as np

from opening_history_stage import OUT, SCRATCH, native, save_json
from opening_history_report import write_csv
from opening_history_cold import feature_file


def main():
    edges=np.exp2(np.arange(-128,8)); bins=len(edges)+2
    ptr=[0]; joint=[]; counts=[]; rows=[]
    with np.load(SCRATCH/'meta.npz') as meta:
        for i,(beta,pid,age,cid) in enumerate(zip(meta['betas'],meta['parent_ids'],meta['parent_age'],meta['ids'])):
            with np.load(feature_file(i)) as f:
                h,o=f['h'],f['op']; code=lambda x: np.where(x==0,0,1+np.searchsorted(edges,x,side='right'))
                ix,n=np.unique(code(h)*bins+code(o),return_counts=True); joint.extend(ix); counts.extend(n);ptr.append(len(joint))
                rows.append(dict(native_row=i,beta=str(beta),parent_id=int(pid),age=int(age),child_id=int(cid),
                    occupied=len(h),mean_Vh=h.mean(),mean_V2O=o.mean(),zero_Vh=int(np.count_nonzero(h==0)),
                    zero_V2O=int(np.count_nonzero(o==0)),positive_opening_zero_h=int(np.count_nonzero((h==0)&(o>0)))))
    write_csv(OUT/'cold-origin-moments.csv',rows)
    np.savez_compressed(OUT/'cold-origin-joint.npz',indptr=np.array(ptr,np.int32),
                        flat_bin=np.array(joint,np.uint16),count=np.array(counts,np.uint16),positive_edges=edges)
    save_json(OUT/'cold-origin-census.json',dict(forks=len(rows),distinct_parents=len({r['parent_id'] for r in rows}),
              occupied_sites=sum(r['occupied'] for r in rows),digest=native.sha(OUT/'cold-origin-joint.npz'),
              columns='CSR per native_row in cold-origin-moments.csv; flat_bin = Vh_bin*n_bins + V2O_bin',
              n_bins=bins,binning='0 exact zero; 1 positive smaller than 2^-128; 2..136 left-closed powers of two; final 137 >= 2^7',
              population='Cold local-all origins only, independent of static release, both ages of each parent; do not pool ages as independent.'))


if __name__=='__main__':main()
