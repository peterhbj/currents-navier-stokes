// D3Q19 lattice Boltzmann (BGK + Smagorinsky LES): escoamento 3D em volta de uma esfera.
// Linhas de corante (streaklines) soltas num plano que passa pelo centro da esfera.
// uso: ./lbm NX NY NZ D Re U steps inject_start snap_every outdir
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static const int cx[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
static const int cy[19] = {0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1};
static const int cz[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1};
static const int opp[19] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
static const float w[19] = {1.f/3,
  1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,
  1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36};

static int NX, NY, NZ;
static size_t N;
#define IDX(x,y,z) (((size_t)(z)*NY + (y))*NX + (x))

static inline float feq(int i, float rho, float ux, float uy, float uz) {
  float cu = 3.f*(cx[i]*ux + cy[i]*uy + cz[i]*uz);
  float uu = 1.5f*(ux*ux + uy*uy + uz*uz);
  return w[i]*rho*(1.f + cu + 0.5f*cu*cu - uu);
}

static float *U3; // ux,uy,uz intercalados
static unsigned char *solid;

static inline void vel_at(float px, float py, float pz, float *o) {
  // interpolação trilinear, y e z periódicos
  if (px < 0) px = 0; if (px > NX-1.001f) px = NX-1.001f;
  int x0 = (int)floorf(px), y0 = (int)floorf(py), z0 = (int)floorf(pz);
  float fx = px-x0, fy = py-y0, fz = pz-z0;
  o[0]=o[1]=o[2]=0;
  for (int dz=0; dz<2; dz++) for (int dy=0; dy<2; dy++) for (int dx=0; dx<2; dx++) {
    int xx = x0+dx; int yy = ((y0+dy)%NY+NY)%NY; int zz = ((z0+dz)%NZ+NZ)%NZ;
    float ww = (dx?fx:1-fx)*(dy?fy:1-fy)*(dz?fz:1-fz);
    const float *u = U3 + 3*IDX(xx,yy,zz);
    o[0]+=ww*u[0]; o[1]+=ww*u[1]; o[2]+=ww*u[2];
  }
}

int main(int argc, char **argv) {
  if (argc < 11) { fprintf(stderr, "uso: NX NY NZ D Re U steps inject_start snap_every outdir\n"); return 1; }
  NX = atoi(argv[1]); NY = atoi(argv[2]); NZ = atoi(argv[3]);
  float D = atof(argv[4]), Re = atof(argv[5]), U = atof(argv[6]);
  int steps = atoi(argv[7]), inj0 = atoi(argv[8]), snapEvery = atoi(argv[9]);
  const char *out = argv[10];
  N = (size_t)NX*NY*NZ;
  float nu = U*D/Re, tau0 = 3.f*nu + 0.5f, Cs = argc>11 ? atof(argv[11]) : 0.12f;
  float scx = NX*0.22f, scy = NY*0.5f, scz = NZ*0.5f, R = D*0.5f;
  printf("N=%zu  nu=%.5f tau=%.4f\n", N, nu, tau0); fflush(stdout);

  float *f = malloc(sizeof(float)*19*N), *g = malloc(sizeof(float)*19*N);
  U3 = calloc(3*N, sizeof(float));
  solid = calloc(N, 1);
  float *tauSponge = malloc(sizeof(float)*NX);
  for (int x=0; x<NX; x++) {
    float s = (float)(x - (NX-40))/40.f; if (s < 0) s = 0;
    tauSponge[x] = tau0 + s*s*(0.8f - tau0 > 0 ? 0.8f - tau0 : 0); // esponja de saída
  }
  for (int z=0; z<NZ; z++) for (int y=0; y<NY; y++) for (int x=0; x<NX; x++) {
    float dx=x-scx, dy=y-scy, dz=z-scz;
    solid[IDX(x,y,z)] = (dx*dx+dy*dy+dz*dz <= R*R);
  }
  unsigned int *nbmask = calloc(N, sizeof(unsigned int));
  #pragma omp parallel for collapse(2)
  for (int z=0; z<NZ; z++) for (int y=0; y<NY; y++) for (int x=1; x<NX-1; x++)
    for (int i=0;i<19;i++) if (solid[IDX(x-cx[i],(y-cy[i]+NY)%NY,(z-cz[i]+NZ)%NZ)]) nbmask[IDX(x,y,z)] |= 1u<<i;
  #pragma omp parallel for
  for (size_t n=0; n<N; n++) for (int i=0;i<19;i++) f[i*N+n] = solid[n]?w[i]:feq(i,1.f,U,0,0);

  // fontes de corante: um "pente" em x=8, plano z = centro da esfera
  float dyS = 1.25f;
  int nsrc = (int)(NY/dyS);
  int injEvery = 4;
  int cap = (int)(1.6f*NX/U/injEvery) + 10;
  float *P = malloc(sizeof(float)*3*nsrc*cap);
  for (size_t k=0;k<(size_t)3*nsrc*cap;k++) P[k]=NAN;
  long injCount = 0;

  char fn[512];
  snprintf(fn,sizeof fn,"%s/forces.txt",out);
  FILE *ff = fopen(fn,"w");
  snprintf(fn,sizeof fn,"%s/meta.txt",out);
  FILE *fm = fopen(fn,"w");
  fprintf(fm,"NX %d\nNY %d\nNZ %d\nD %g\nRe %g\nU %g\nscx %g\nscy %g\nscz %g\nnsrc %d\ncap %d\ninjEvery %d\ndyS %g\n",NX,NY,NZ,D,Re,U,scx,scy,scz,nsrc,cap,injEvery,dyS);
  fclose(fm);

  double t0 = omp_get_wtime();
  for (int t=0; t<steps; t++) {
    double Fx=0,Fy=0,Fz=0;
    #pragma omp parallel for collapse(2) reduction(+:Fx,Fy,Fz) schedule(static)
    for (int z=0; z<NZ; z++) for (int y=0; y<NY; y++) {
      size_t rowsrc[19];
      for (int i=0;i<19;i++) rowsrc[i] = IDX(0,(y-cy[i]+NY)%NY,(z-cz[i]+NZ)%NZ) - cx[i];
      static __thread float F[19][1024], FE[19][1024];
      float rho[1024], ux[1024], uy[1024], uz[1024], Q[1024];
      size_t nb = IDX(0,y,z);
      for (int i=0;i<19;i++) memcpy(&F[i][1], f + i*N + rowsrc[i] + 1, sizeof(float)*(NX-2));
      for (int x=1; x<NX-1; x++) {
        unsigned int msk = nbmask[nb+x];
        if (msk) for (int i=0;i<19;i++) if (msk & (1u<<i)) {
          float v = f[opp[i]*N+nb+x];
          F[i][x] = v;
          Fx += -2.0*cx[i]*v; Fy += -2.0*cy[i]*v; Fz += -2.0*cz[i]*v;
        }
      }
      const int X0=1, X1=NX-1;
      for (int x=X0;x<X1;x++){rho[x]=0;ux[x]=0;uy[x]=0;uz[x]=0;Q[x]=0;}
      for (int i=0;i<19;i++){
        const float ccx=cx[i], ccy=cy[i], ccz=cz[i];
        for (int x=X0;x<X1;x++){float v=F[i][x]; rho[x]+=v; ux[x]+=ccx*v; uy[x]+=ccy*v; uz[x]+=ccz*v;}
      }
      for (int x=X0;x<X1;x++){float r=1.f/rho[x]; ux[x]*=r; uy[x]*=r; uz[x]*=r;}
      float Pxx[1024],Pyy[1024],Pzz[1024],Pxy[1024],Pxz[1024],Pyz[1024];
      for (int x=X0;x<X1;x++){Pxx[x]=Pyy[x]=Pzz[x]=Pxy[x]=Pxz[x]=Pyz[x]=0;}
      for (int i=0;i<19;i++){
        const float ccx=cx[i], ccy=cy[i], ccz=cz[i], wi=w[i];
        for (int x=X0;x<X1;x++){
          float cu=3.f*(ccx*ux[x]+ccy*uy[x]+ccz*uz[x]);
          float uu=1.5f*(ux[x]*ux[x]+uy[x]*uy[x]+uz[x]*uz[x]);
          float fe=wi*rho[x]*(1.f+cu+0.5f*cu*cu-uu);
          FE[i][x]=fe; float d=F[i][x]-fe;
          Pxx[x]+=ccx*ccx*d; Pyy[x]+=ccy*ccy*d; Pzz[x]+=ccz*ccz*d;
          Pxy[x]+=ccx*ccy*d; Pxz[x]+=ccx*ccz*d; Pyz[x]+=ccy*ccz*d;
        }
      }
      for (int x=X0;x<X1;x++){
        float q=sqrtf(Pxx[x]*Pxx[x]+Pyy[x]*Pyy[x]+Pzz[x]*Pzz[x]+2.f*(Pxy[x]*Pxy[x]+Pxz[x]*Pxz[x]+Pyz[x]*Pyz[x]));
        float tb=tauSponge[x];
        Q[x]=2.f/(tb+sqrtf(tb*tb+18.f*1.41421356f*Cs*Cs*q/rho[x]));
      }
      for (int i=0;i<19;i++){
        float *gi = g + i*N + nb;
        for (int x=X0;x<X1;x++) gi[x]=F[i][x]-Q[x]*(F[i][x]-FE[i][x]);
      }
      for (int x=X0;x<X1;x++){
        size_t n=nb+x;
        if (solid[n]) { U3[3*n]=U3[3*n+1]=U3[3*n+2]=0; for (int i=0;i<19;i++) g[i*N+n]=w[i]; }
        else { U3[3*n]=ux[x]; U3[3*n+1]=uy[x]; U3[3*n+2]=uz[x]; }
      }
      // entrada (x=0): equilíbrio com velocidade U; saída: gradiente zero
      size_t n0 = IDX(0,y,z), n1 = IDX(NX-1,y,z), n2 = IDX(NX-2,y,z);
      float pv = t < 3000 ? 0.02f*U*sinf(6.2831853f*t/700.f) : 0.f;
      float pw = t < 3000 ? 0.012f*U*sinf(6.2831853f*t/530.f + 1.f) : 0.f;
      for (int i=0;i<19;i++) g[i*N+n0] = feq(i,1.f,U,pv,pw);
      U3[3*n0]=U;U3[3*n0+1]=pv;U3[3*n0+2]=pw;
    }
    #pragma omp parallel for collapse(2)
    for (int z=0; z<NZ; z++) for (int y=0; y<NY; y++) {
      size_t n1 = IDX(NX-1,y,z), n2 = IDX(NX-2,y,z);
      for (int i=0;i<19;i++) g[i*N+n1] = g[i*N+n2];
      for (int k=0;k<3;k++) U3[3*n1+k]=U3[3*n2+k];
    }
    float *tmp=f; f=g; g=tmp;

    if (t % 10 == 0) {
      fprintf(ff,"%d %.6e %.6e %.6e\n",t,Fx,Fy,Fz);
      if (t % 500 == 0) {
        double el = omp_get_wtime()-t0;
        float Cd = Fx/(0.5f*U*U*M_PI*R*R);
        printf("t=%d  Cd=%.3f  MLUPS=%.1f\n", t, Cd, (double)N*(t+1)/el/1e6); fflush(stdout);
        fflush(ff);
      }
    }

    // corante
    if (t >= inj0) {
      if ((t - inj0) % injEvery == 0) {
        int k = injCount % cap;
        for (int s=0;s<nsrc;s++) {
          float *p = P + 3*((size_t)s*cap + k);
          p[0]=8.f; p[1]=(s+0.5f)*dyS + 0.013f; p[2]=scz;
        }
        injCount++;
      }
      #pragma omp parallel for schedule(static)
      for (size_t q=0; q<(size_t)nsrc*cap; q++) {
        float *p = P+3*q;
        if (isnan(p[0])) continue;
        float u1[3], u2[3];
        vel_at(p[0],p[1],p[2],u1);
        vel_at(p[0]+0.5f*u1[0],p[1]+0.5f*u1[1],p[2]+0.5f*u1[2],u2);
        p[0]+=u2[0]; p[1]+=u2[1]; p[2]+=u2[2];
        if (p[1]<0) p[1]+=NY; if (p[1]>=NY) p[1]-=NY;
        if (p[2]<0) p[2]+=NZ; if (p[2]>=NZ) p[2]-=NZ;
        if (p[0] > NX-3) p[0]=p[1]=p[2]=NAN;
      }
      if (snapEvery > 0 && (t - inj0) % snapEvery == 0 && t > inj0) {
        snprintf(fn,sizeof fn,"%s/snap_%06d.bin",out,t);
        FILE *fs=fopen(fn,"wb");
        long ic = injCount;
        fwrite(&ic,sizeof ic,1,fs);
        fwrite(P,sizeof(float),3*(size_t)nsrc*cap,fs);
        fclose(fs);
      }
    }
  }
  // campo final: plano z=centro e plano y=centro
  snprintf(fn,sizeof fn,"%s/planeZ.bin",out);
  FILE *fp=fopen(fn,"wb");
  int zc=(int)scz;
  for (int y=0;y<NY;y++) fwrite(U3+3*IDX(0,y,zc),sizeof(float),3*NX,fp);
  fclose(fp);
  snprintf(fn,sizeof fn,"%s/planeY.bin",out);
  fp=fopen(fn,"wb");
  int yc=(int)scy;
  for (int z=0;z<NZ;z++) fwrite(U3+3*IDX(0,yc,z),sizeof(float),3*NX,fp);
  fclose(fp);
  fclose(ff);
  printf("pronto em %.1fs\n", omp_get_wtime()-t0);
  return 0;
}
