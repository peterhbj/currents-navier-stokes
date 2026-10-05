import sys, glob, numpy as np
sys.argv=['x','re1000']
exec(open('render.py').read().split('if __name__')[0])
snaps = sorted(glob.glob('re1000/snap_*.bin'))[-20:]
ks = [-2,-1,-0.5,0,0.5,1,2,4,6,8]
res = {k: [] for k in ks}
for fn in snaps:
    P = load(fn)
    for s in range(nsrc):
        y0 = (s+0.5)*meta['dyS']
        p = P[s][~np.isnan(P[s][:,0])]
        for k in ks:
            xs = scx + k*D
            m = np.abs(p[:,0]-xs) < 0.6
            if m.any():
                dev = np.sqrt((p[m,1]-y0)**2 + (p[m,2]-scz)**2).max()
                res[k].append((abs(y0-scy), dev))
print("x/D   largura lateral perturbada (desvio>0.1D), em diâmetros D")
for k in ks:
    a = np.array(res[k])
    dist = a[a[:,1] > 0.1*D, 0]
    w = 2*np.percentile(dist, 95)/D if len(dist) else 0
    print(f"{k:5.1f}  {w:.2f}")
