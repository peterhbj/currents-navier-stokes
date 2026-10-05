import numpy as np
from PIL import Image
from scipy import ndimage as nd
im = np.asarray(Image.open("cover.jpg").convert("L"), np.float32)/255
gx = nd.sobel(im, 1); gy = nd.sobel(im, 0)
def coh(sig):
    Jxx = nd.gaussian_filter(gx*gx, sig); Jyy = nd.gaussian_filter(gy*gy, sig); Jxy = nd.gaussian_filter(gx*gy, sig)
    tr = Jxx+Jyy+1e-6
    return np.sqrt((Jxx-Jyy)**2+4*Jxy**2)/tr, 0.5*np.arctan2(2*Jxy, Jxx-Jyy)
c, th = coh(6)
# desvio de orientação em relação à orientação de grande escala das linhas retas
_, thL = coh(60)
dev = np.abs(np.angle(np.exp(2j*(th-thL))))/2
dist = (1-c)*0.5 + dev  # índice de "perturbação"
dist[im < 0.06] = 0     # fundo preto
dist = nd.gaussian_filter(dist, 4)
v = np.clip(dist/0.6, 0, 1)
rgb = np.asarray(Image.open("cover.jpg").convert("RGB"), np.float32)*0.35
heat = np.stack([255*v, 200*v**2, 40*v], -1)
Image.fromarray(np.clip(rgb+heat,0,255).astype(np.uint8)).save("cover_heat.png")
# perfil: perturbação média por coluna na linha do centro da bola e acima (montante)
np.save("cover_dist.npy", v)
for yrow in (300, 400, 480, 560, 650, 720):
    row = v[yrow]
    print(yrow, " ".join(f"{row[x]:.2f}" for x in range(0,1500,100)))
