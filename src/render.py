import sys, glob, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

run = sys.argv[1]
meta = {l.split()[0]: float(l.split()[1]) for l in open(f"{run}/meta.txt")}
NX, NY, NZ = int(meta["NX"]), int(meta["NY"]), int(meta["NZ"])
nsrc, cap = int(meta["nsrc"]), int(meta["cap"])
D, U = meta["D"], meta["U"]
scx, scy, scz = meta["scx"], meta["scy"], meta["scz"]
R = D / 2


def load(fn):
    with open(fn, "rb") as fh:
        ic = np.frombuffer(fh.read(8), np.int64)[0]
        P = np.frombuffer(fh.read(), np.float32).reshape(nsrc, cap, 3)
    order = (ic + np.arange(cap)) % cap  # do mais velho pro mais novo
    return P[:, order, :]


def polylines(P, maxgap=12.0):
    """quebra cada streakline em trechos contínuos"""
    out = []
    for s in range(P.shape[0]):
        p = P[s]
        alive = ~np.isnan(p[:, 0])
        p = p[alive]
        if len(p) < 2:
            continue
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        cuts = np.where(d > maxgap)[0] + 1
        pieces = np.split(p, cuts)
        if pieces[-1][-1, 0] < 12:  # escoamento uniforme a montante do pente: prolonga reto
            q = pieces[-1][-1]; xe = np.linspace(q[0] - 2, -60 * D, 120)
            pieces[-1] = np.vstack([pieces[-1], np.c_[xe, np.full_like(xe, q[1]), np.full_like(xe, q[2])]])
        for seg in pieces:
            if len(seg) > 1:
                out.append((s, seg))
    return out


def line_color(s, t=0.0):
    center = int(scy / meta["dyS"])
    if abs(s - center) == 0:
        return (255, 60, 70)
    # lavanda/roxo alternado, como a capa
    base = np.array([196, 150, 235]) if s % 2 == 0 else np.array([150, 100, 210])
    return tuple(int(c) for c in base)


# ---------- vista de cima ----------
def top_view(P, fn, S=4):
    W, H = NX * S, NY * S
    img = Image.new("RGB", (W * 2, H * 2), (8, 4, 14))
    dr = ImageDraw.Draw(img)
    k = 2 * S
    for s, seg in polylines(P):
        pts = [(float(x * k), float(y * k)) for x, y in seg[:, :2]]
        dr.line(pts, fill=line_color(s), width=max(2, int(0.5 * k)))
    cxp, cyp, r = scx * k, scy * k, R * k
    dr.ellipse([cxp - r, cyp - r, cxp + r, cyp + r], fill=(170, 170, 178), outline=(230, 230, 235), width=3)
    img = img.resize((W, H), Image.LANCZOS)
    img.save(fn)


# ---------- vista em perspectiva ("câmera da capa") ----------
CAM = [9.5, -1.6, 2.4, -14, 1.1, 1.25]


def perspective(P, fn, W=1500, H=750):
    # câmera a jusante, elevada, olhando contra o escoamento (como na capa)
    cam = np.array([scx + CAM[0] * D, scy + CAM[1] * D, scz + CAM[2] * D])
    tgt = np.array([scx + CAM[3] * D, scy + CAM[4] * D, scz])
    fwd = tgt - cam; fwd /= np.linalg.norm(fwd)
    up0 = np.array([0, 0, 1.0])
    right = np.cross(fwd, up0); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = CAM[5] * W
    ss = 2

    def proj(p):
        v = p - cam
        zc = v @ fwd
        xs = (v @ right) / zc * f + 0.5 * W
        ys = -(v @ up) / zc * f + 0.5 * H
        return xs * ss, ys * ss, zc

    img = Image.new("RGB", (W * ss, H * ss), (6, 3, 10))
    dr = ImageDraw.Draw(img)
    segs = polylines(P)
    # fora do domínio lateral o escoamento é uniforme (perturbação potencial < 1%): linhas retas de preenchimento
    xe = np.linspace(-60 * D, NX - 3, 400)
    for k in range(1, int(10 * D / meta["dyS"])):
        for yy in (-k * meta["dyS"] + 0.5 * meta["dyS"], NY + (k - 0.5) * meta["dyS"]):
            segs.append((k + nsrc, np.c_[xe, np.full_like(xe, yy), np.full_like(xe, scz)]))
    # pinta do mais longe pro mais perto
    items = []
    for s, seg in segs:
        xs, ys, zc = proj(seg.astype(np.float64))
        ok = zc > 1
        xs, ys, zc = xs[ok], ys[ok], zc[ok]
        for a in range(0, len(xs) - 1, 12):  # pedaços curtos -> ordem de profundidade correta
            b = min(a + 13, len(xs))
            if b - a >= 2:
                items.append((zc[a:b].mean(), s, xs[a:b], ys[a:b], zc[a:b]))
    items.sort(key=lambda t: -t[0])
    # esfera
    sc = np.array([scx, scy, scz])
    bx, by, bz = proj(sc[None])
    br = R / bz[0] * f * ss
    drawn_ball = False
    for depth, s, xs, ys, zc in items:
        if not drawn_ball and depth < bz[0]:
            draw_ball(img, bx[0], by[0], br); dr = ImageDraw.Draw(img); drawn_ball = True
        fog = np.clip(1.15 - (depth / (60 * D)), 0.25, 1.0)
        c = tuple(int(v * fog) for v in line_color(s))
        wdt = max(1, int(round(2.6 * ss * (5 * D) / max(depth, 1) * 0.6)))
        dr.line(list(zip(xs.tolist(), ys.tolist())), fill=c, width=wdt)
    if not drawn_ball:
        draw_ball(img, bx[0], by[0], br)
    img = img.resize((W, H), Image.LANCZOS)
    img.save(fn)


def draw_ball(img, x, y, r):
    n = int(2 * r) + 4
    yy, xx = np.mgrid[0:n, 0:n] - n / 2
    d2 = (xx ** 2 + yy ** 2) / r ** 2
    mask = d2 <= 1
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    nx_, ny_ = xx / r, yy / r
    L = np.array([-0.4, -0.6, 0.7]); L /= np.linalg.norm(L)
    diff = np.clip(nx_ * L[0] + ny_ * L[1] + nz * L[2], 0, 1)
    spec = np.clip(nx_ * 0.25 + ny_ * 0.45 + nz * 0.86, 0, 1) ** 60
    env = 0.25 + 0.35 * (1 - nz) * (ny_ > 0)  # reflexo do chão roxo
    col = np.stack([40 + 150 * diff + 255 * spec + 60 * env,
                    40 + 150 * diff + 255 * spec + 30 * env,
                    45 + 155 * diff + 255 * spec + 80 * env], -1)
    col = np.clip(col, 0, 255).astype(np.uint8)
    a = (mask * 255).astype(np.uint8)
    sprite = Image.fromarray(col, "RGB")
    alpha = Image.fromarray(a, "L").filter(ImageFilter.GaussianBlur(1))
    img.paste(sprite, (int(x - n / 2), int(y - n / 2)), alpha)


# ---------- vorticidade no plano z = centro ----------
def vorticity(fn):
    u = np.fromfile(f"{run}/planeZ.bin", np.float32).reshape(NY, NX, 3)
    wz = np.gradient(u[..., 1], axis=1) - np.gradient(u[..., 0], axis=0)
    v = np.clip(wz / (3 * U / D), -1, 1)
    # mapa divergente roxo-preto-laranja
    r = np.where(v > 0, 255 * v, 120 * -v)
    g = np.where(v > 0, 140 * v, 60 * -v)
    b = np.where(v > 0, 40 * v, 255 * -v)
    img = np.stack([r, g, b], -1).astype(np.uint8)
    yy, xx = np.mgrid[0:NY, 0:NX]
    img[(xx - scx) ** 2 + (yy - scy) ** 2 <= R * R] = (200, 200, 205)
    Image.fromarray(img).resize((NX * 4, NY * 4), Image.LANCZOS).save(fn)


# ---------- forças ----------
def forces():
    a = np.loadtxt(f"{run}/forces.txt")
    t, Fx, Fy, Fz = a.T
    q = 0.5 * U * U * np.pi * R * R
    m = t > t.max() * 0.45
    Cd = Fx[m].mean() / q
    out = {"Cd": Cd, "Cd_std": Fx[m].std() / q}
    for name, F in (("y", Fy), ("z", Fz)):
        s = F[m] - F[m].mean()
        sp = np.abs(np.fft.rfft(s * np.hanning(len(s))))
        fr = np.fft.rfftfreq(len(s), d=10.0)
        k = sp[1:].argmax() + 1
        out["St_" + name] = fr[k] * D / U
        out["Cl_rms_" + name] = s.std() / q
    np.savetxt(f"{run}/cd_series.txt", np.c_[t, Fx / q, Fy / q, Fz / q], fmt="%.5f")
    return out


if __name__ == "__main__":
    snaps = sorted(glob.glob(f"{run}/snap_*.bin"))
    last = load(snaps[-1])
    top_view(last, f"{run}/top.png")
    perspective(last, f"{run}/persp.png")
    vorticity(f"{run}/vort.png")
    print(forces())
    if "--anim" in sys.argv:
        os.makedirs(f"{run}/frames", exist_ok=True)
        for i, fn in enumerate(snaps):
            perspective(load(fn), f"{run}/frames/p_{i:04d}.png", W=1200, H=600)
            top_view(load(fn), f"{run}/frames/t_{i:04d}.png", S=3)
