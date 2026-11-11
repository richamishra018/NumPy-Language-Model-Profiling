/*
 * tx.c : C + OpenBLAS port of transformer.py (float64, row-major).
 * Forward, cross-entropy loss, backward and Adam, with a built-in check against
 * NumPy reference files written by baseline/export_ref.py.
 *
 * Build : gcc -O3 -march=native -g -fno-omit-frame-pointer -o tx tx.c -lopenblas -lm
 * Check : ./tx check  ref/
 * Bench : ./tx bench  ref/ 50      (50 timed repetitions per phase)
 * Train : ./tx train  ref/         (replays the exported batches)
 */
#define _POSIX_C_SOURCE 200809L
#include <cblas.h>
#ifdef USE_ITT                       /* optional VTune markers: build with -DUSE_ITT -I<ittapi>/include and link -littnotify */
#include <ittnotify.h>
static __itt_domain *dom;
static __itt_string_handle *sh_fwd, *sh_loss, *sh_bwd, *sh_adam;
#define ITT_BEGIN(h) __itt_task_begin(dom, __itt_null, __itt_null, h)
#define ITT_END()    __itt_task_end(dom)
#else
#define ITT_BEGIN(h) ((void)0)
#define ITT_END()    ((void)0)
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------- configuration and tensors ---------- */
typedef struct { int V, S, D, H, F, L, B, steps; double lr; } Cfg;

typedef struct {               /* pointers into one flat parameter block */
    double *tok, *pos;
    struct { double *Wq,*Wk,*Wv,*Wo,*bq,*bk,*bv,*bo,*ln1g,*ln1b,*ln2g,*ln2b,*W1,*b1,*W2,*b2; } l[16];
    double *lnfg, *lnfb, *Wout, *bout;
    size_t n;                  /* total number of doubles */
} Params;

static void layout(Params *p, double *base, const Cfg *c) {
    size_t o = 0; int D = c->D, F = c->F, V = c->V;
    p->tok = base + o; o += (size_t)V * D;
    p->pos = base + o; o += (size_t)(c->S + 1) * D;
    for (int l = 0; l < c->L; l++) {
        p->l[l].Wq = base + o; o += (size_t)D*D;  p->l[l].Wk = base + o; o += (size_t)D*D;
        p->l[l].Wv = base + o; o += (size_t)D*D;  p->l[l].Wo = base + o; o += (size_t)D*D;
        p->l[l].bq = base + o; o += D; p->l[l].bk = base + o; o += D;
        p->l[l].bv = base + o; o += D; p->l[l].bo = base + o; o += D;
        p->l[l].ln1g = base + o; o += D; p->l[l].ln1b = base + o; o += D;
        p->l[l].ln2g = base + o; o += D; p->l[l].ln2b = base + o; o += D;
        p->l[l].W1 = base + o; o += (size_t)D*F;  p->l[l].b1 = base + o; o += F;
        p->l[l].W2 = base + o; o += (size_t)F*D;  p->l[l].b2 = base + o; o += D;
    }
    p->lnfg = base + o; o += D; p->lnfb = base + o; o += D;
    p->Wout = base + o; o += (size_t)D*V; p->bout = base + o; o += V;
    p->n = o;
}

/* activations saved by forward() for backward() */
typedef struct {
    double *x_in[16], *xhat1[16], *rstd1[16], *xn[16], *Q[16], *K[16], *Vv[16],
           *attn[16], *ctx[16], *x_mid[16], *xhat2[16], *rstd2[16], *xn2[16],
           *h[16], *hact[16], *x_out[16];
    double *xhatf, *rstdf, *xf, *logits, *x0;
} Acts;

static Cfg C; static Params P, G; static Acts A;
static double *pbuf, *gbuf, *mbuf, *vbuf;
static int adam_t = 0;

static double *zalloc(size_t n) { double *p = calloc(n, sizeof(double)); if (!p) { perror("calloc"); exit(1);} return p; }
#ifdef _WIN32
#include <windows.h>
static double now(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart / (double)f.QuadPart; }
#else
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
#endif

/* ---------- elementwise kernels ---------- */
static const double K0 = 0.7978845608028654;   /* sqrt(2/pi) */
static void gelu_fwd(const double *h, double *o, size_t n) {
    for (size_t i = 0; i < n; i++) { double x = h[i]; o[i] = 0.5*x*(1.0 + tanh(K0*(x + 0.044715*x*x*x))); }
}
static void gelu_bwd(const double *h, const double *dact, double *dh, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double x = h[i], t = tanh(K0*(x + 0.044715*x*x*x));
        double g = 0.5*(1.0 + t) + 0.5*x*(1.0 - t*t)*K0*(1.0 + 3*0.044715*x*x);
        dh[i] = dact[i] * g;
    }
}
static void add_bias(double *y, const double *b, int N, int D) {
    for (int i = 0; i < N; i++) for (int j = 0; j < D; j++) y[(size_t)i*D + j] += b[j];
}
static void colsum_acc(const double *dy, double *db, int N, int D) {
    for (int i = 0; i < N; i++) for (int j = 0; j < D; j++) db[j] += dy[(size_t)i*D + j];
}
/* y(N x Dout) = x(N x Din) @ W(Din x Dout) + b */
static void linear(const double *x, const double *W, const double *b, double *y, int N, int Din, int Dout) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, Dout, Din, 1.0, x, Din, W, Dout, 0.0, y, Dout);
    add_bias(y, b, N, Dout);
}

static void ln_fwd(const double *x, const double *g, const double *b, double *out, double *xhat, double *rstd, int N, int D) {
    for (int i = 0; i < N; i++) {
        const double *xi = x + (size_t)i*D; double m = 0, v = 0;
        for (int j = 0; j < D; j++) m += xi[j];
        m /= D;
        for (int j = 0; j < D; j++) { double d = xi[j] - m; v += d*d; }
        v /= D;
        double r = 1.0 / sqrt(v + 1e-5); rstd[i] = r;
        for (int j = 0; j < D; j++) { double xh = (xi[j] - m) * r; xhat[(size_t)i*D + j] = xh; out[(size_t)i*D + j] = g[j]*xh + b[j]; }
    }
}
/* dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)); dg, db accumulated */
static void ln_bwd(const double *dout, const double *xhat, const double *rstd, const double *g,
                   double *dx, double *dg, double *db, int N, int D) {
    for (int i = 0; i < N; i++) {
        const double *d = dout + (size_t)i*D, *xh = xhat + (size_t)i*D; double s1 = 0, s2 = 0;
        for (int j = 0; j < D; j++) { double dxh = d[j]*g[j]; s1 += dxh; s2 += dxh*xh[j]; dg[j] += d[j]*xh[j]; db[j] += d[j]; }
        s1 /= D; s2 /= D;
        for (int j = 0; j < D; j++) dx[(size_t)i*D + j] = rstd[i] * (d[j]*g[j] - s1 - xh[j]*s2);
    }
}

/* ---------- forward ---------- */
static void forward(const int *ids /* B x (S+1) */) {
    int B = C.B, T = C.S + 1, D = C.D, H = C.H, F = C.F, N = B*T, dh = D/H;   /* NumPy forward() sees the full seq_len+1 row */
    double scale = 1.0 / sqrt((double)dh);
    for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) {
        const double *te = P.tok + (size_t)ids[b*T + t]*D, *pe = P.pos + (size_t)t*D;
        double *x = A.x0 + ((size_t)b*T + t)*D;
        for (int j = 0; j < D; j++) x[j] = te[j] + pe[j];
    }
    const double *x = A.x0;
    for (int l = 0; l < C.L; l++) {
        A.x_in[l] = (double*)x;
        ln_fwd(x, P.l[l].ln1g, P.l[l].ln1b, A.xn[l], A.xhat1[l], A.rstd1[l], N, D);
        linear(A.xn[l], P.l[l].Wq, P.l[l].bq, A.Q[l], N, D, D);
        linear(A.xn[l], P.l[l].Wk, P.l[l].bk, A.K[l], N, D, D);
        linear(A.xn[l], P.l[l].Wv, P.l[l].bv, A.Vv[l], N, D, D);
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            size_t off = ((size_t)b*T)*D + (size_t)h*dh;             /* strided head view, ld = D */
            double *S = A.attn[l] + ((size_t)(b*H + h))*T*T;
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, T, dh, scale, A.Q[l]+off, D, A.K[l]+off, D, 0.0, S, T);
            for (int i = 0; i < T; i++) {
                double *r = S + (size_t)i*T, mx = -1e300, sum = 0;
                for (int j = i+1; j < T; j++) r[j] = -1e9;           /* causal mask */
                for (int j = 0; j < T; j++) if (r[j] > mx) mx = r[j];
                for (int j = 0; j < T; j++) { r[j] = exp(r[j] - mx); sum += r[j]; }
                for (int j = 0; j < T; j++) r[j] /= sum;
            }
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, dh, T, 1.0, S, T, A.Vv[l]+off, D, 0.0, A.ctx[l]+off, D);
        }
        double *xm = A.x_mid[l];
        linear(A.ctx[l], P.l[l].Wo, P.l[l].bo, xm, N, D, D);
        for (size_t i = 0; i < (size_t)N*D; i++) xm[i] += x[i];       /* residual */
        ln_fwd(xm, P.l[l].ln2g, P.l[l].ln2b, A.xn2[l], A.xhat2[l], A.rstd2[l], N, D);
        linear(A.xn2[l], P.l[l].W1, P.l[l].b1, A.h[l], N, D, F);
        gelu_fwd(A.h[l], A.hact[l], (size_t)N*F);
        double *xo = A.x_out[l];
        linear(A.hact[l], P.l[l].W2, P.l[l].b2, xo, N, F, D);
        for (size_t i = 0; i < (size_t)N*D; i++) xo[i] += xm[i];      /* residual */
        x = xo;
    }
    ln_fwd(x, P.lnfg, P.lnfb, A.xf, A.xhatf, A.rstdf, N, D);
    linear(A.xf, P.Wout, P.bout, A.logits, N, D, C.V);
}

/* ---------- loss: returns loss, writes dlogits (N x V) ---------- */
static double loss_and_dlogits(const int *ids, double *dl) {
    int B = C.B, T = C.S + 1, V = C.V; double loss = 0; double inv = 1.0 / ((double)B*(T-1));
    memset(dl, 0, (size_t)B*T*V*sizeof(double));
    for (int b = 0; b < B; b++) for (int t = 0; t < T-1; t++) {
        const double *lg = A.logits + ((size_t)b*T + t)*V; double *d = dl + ((size_t)b*T + t)*V;
        double mx = -1e300, sum = 0; int tgt = ids[b*T + t + 1];
        for (int j = 0; j < V; j++) if (lg[j] > mx) mx = lg[j];
        for (int j = 0; j < V; j++) { d[j] = exp(lg[j] - mx); sum += d[j]; }
        for (int j = 0; j < V; j++) d[j] /= sum;
        loss += -log(d[tgt] + 1e-9);
        d[tgt] -= 1.0;
        for (int j = 0; j < V; j++) d[j] *= inv;
    }
    return loss * inv;
}

/* ---------- backward ---------- */
static void backward(const int *ids, const double *dlogits) {
    int B = C.B, T = C.S + 1, D = C.D, H = C.H, F = C.F, V = C.V, N = B*T, dh = D/H;
    double scale = 1.0 / sqrt((double)dh);
    memset(gbuf, 0, G.n * sizeof(double));
    double *dx   = zalloc((size_t)N*D), *tmp = zalloc((size_t)N*D), *dxn = zalloc((size_t)N*D);
    double *dact = zalloc((size_t)N*F), *dhh = zalloc((size_t)N*F);
    double *dctx = zalloc((size_t)N*D), *dQ = zalloc((size_t)N*D), *dK = zalloc((size_t)N*D), *dV = zalloc((size_t)N*D);
    double *dattn = zalloc((size_t)T*T), *dsc = zalloc((size_t)T*T);

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, V, N, 1.0, A.xf, D, dlogits, V, 0.0, G.Wout, V);
    colsum_acc(dlogits, G.bout, N, V);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, D, V, 1.0, dlogits, V, P.Wout, V, 0.0, dx, D);
    ln_bwd(dx, A.xhatf, A.rstdf, P.lnfg, tmp, G.lnfg, G.lnfb, N, D);
    memcpy(dx, tmp, (size_t)N*D*sizeof(double));

    for (int l = C.L - 1; l >= 0; l--) {
        /* feed-forward branch (dx is both the residual path and the branch gradient) */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, F, D, 1.0, dx, D, P.l[l].W2, D, 0.0, dact, F);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, F, D, N, 1.0, A.hact[l], F, dx, D, 0.0, G.l[l].W2, D);
        colsum_acc(dx, G.l[l].b2, N, D);
        gelu_bwd(A.h[l], dact, dhh, (size_t)N*F);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, F, N, 1.0, A.xn2[l], D, dhh, F, 0.0, G.l[l].W1, F);
        colsum_acc(dhh, G.l[l].b1, N, F);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, D, F, 1.0, dhh, F, P.l[l].W1, F, 0.0, dxn, D);
        ln_bwd(dxn, A.xhat2[l], A.rstd2[l], P.l[l].ln2g, tmp, G.l[l].ln2g, G.l[l].ln2b, N, D);
        for (size_t i = 0; i < (size_t)N*D; i++) dx[i] += tmp[i];

        /* attention branch */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, D, D, 1.0, dx, D, P.l[l].Wo, D, 0.0, dctx, D);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, D, N, 1.0, A.ctx[l], D, dx, D, 0.0, G.l[l].Wo, D);
        colsum_acc(dx, G.l[l].bo, N, D);
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
            size_t off = ((size_t)b*T)*D + (size_t)h*dh;
            const double *At = A.attn[l] + ((size_t)(b*H + h))*T*T;
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, T, dh, T, 1.0, At, T, dctx+off, D, 0.0, dV+off, D);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, T, dh, 1.0, dctx+off, D, A.Vv[l]+off, D, 0.0, dattn, T);
            for (int i = 0; i < T; i++) {
                const double *a = At + (size_t)i*T, *da = dattn + (size_t)i*T; double s = 0;
                for (int j = 0; j < T; j++) s += da[j]*a[j];
                for (int j = 0; j < T; j++) dsc[(size_t)i*T + j] = (j > i) ? 0.0 : a[j]*(da[j] - s)*scale;
            }
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, dh, T, 1.0, dsc, T, A.K[l]+off, D, 0.0, dQ+off, D);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, T, dh, T, 1.0, dsc, T, A.Q[l]+off, D, 0.0, dK+off, D);
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, D, D, 1.0, dQ, D, P.l[l].Wq, D, 0.0, dxn, D);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, D, D, 1.0, dK, D, P.l[l].Wk, D, 1.0, dxn, D);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, D, D, 1.0, dV, D, P.l[l].Wv, D, 1.0, dxn, D);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, D, N, 1.0, A.xn[l], D, dQ, D, 0.0, G.l[l].Wq, D);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, D, N, 1.0, A.xn[l], D, dK, D, 0.0, G.l[l].Wk, D);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, D, D, N, 1.0, A.xn[l], D, dV, D, 0.0, G.l[l].Wv, D);
        colsum_acc(dQ, G.l[l].bq, N, D); colsum_acc(dK, G.l[l].bk, N, D); colsum_acc(dV, G.l[l].bv, N, D);
        ln_bwd(dxn, A.xhat1[l], A.rstd1[l], P.l[l].ln1g, tmp, G.l[l].ln1g, G.l[l].ln1b, N, D);
        for (size_t i = 0; i < (size_t)N*D; i++) dx[i] += tmp[i];
    }
    for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) {
        const double *d = dx + ((size_t)b*T + t)*D;
        double *gt = G.tok + (size_t)ids[b*T + t]*D, *gp = G.pos + (size_t)t*D;
        for (int j = 0; j < D; j++) { gt[j] += d[j]; gp[j] += d[j]; }
    }
    free(dx); free(tmp); free(dxn); free(dact); free(dhh); free(dctx); free(dQ); free(dK); free(dV); free(dattn); free(dsc);
}

/* ---------- Adam (matches AdamOptimizer: coupled L2 decay on 2-D tensors) ---------- */
static void adam_step(void) {
    adam_t++;
    double b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0.01;
    double c1 = 1.0 - pow(b1, adam_t), c2 = 1.0 - pow(b2, adam_t);
    /* decay applies only to tensors with ndim >= 2: tok, pos, W*, Wout. Mark them by ranges. */
    size_t n = P.n;
    char *dec = calloc(n, 1);
    #define MARK(ptr, count) memset(dec + ((ptr) - pbuf), 1, (size_t)(count))
    MARK(P.tok, (size_t)C.V*C.D); MARK(P.pos, (size_t)(C.S+1)*C.D);
    for (int l = 0; l < C.L; l++) {
        MARK(P.l[l].Wq, (size_t)C.D*C.D); MARK(P.l[l].Wk, (size_t)C.D*C.D);
        MARK(P.l[l].Wv, (size_t)C.D*C.D); MARK(P.l[l].Wo, (size_t)C.D*C.D);
        MARK(P.l[l].W1, (size_t)C.D*C.F); MARK(P.l[l].W2, (size_t)C.F*C.D);
    }
    MARK(P.Wout, (size_t)C.D*C.V);
    for (size_t i = 0; i < n; i++) {
        double g = gbuf[i] + (dec[i] ? wd * pbuf[i] : 0.0);
        mbuf[i] = b1*mbuf[i] + (1-b1)*g;
        vbuf[i] = b2*vbuf[i] + (1-b2)*g*g;
        pbuf[i] -= C.lr * (mbuf[i]/c1) / (sqrt(vbuf[i]/c2) + eps);
    }
    free(dec);
}

/* ---------- helpers ---------- */
static void *slurp(const char *dir, const char *name, size_t *bytes) {
    char path[512]; snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(n); if (fread(buf, 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f); if (bytes) *bytes = n; return buf;
}
static double maxdiff(const double *a, const double *b, size_t n) {
    double m = 0; for (size_t i = 0; i < n; i++) { double d = fabs(a[i]-b[i]); if (d > m) m = d; } return m;
}

static void alloc_acts(void) {
    int T = C.S + 1, N = C.B*T, D = C.D, F = C.F, H = C.H;
    A.x0 = zalloc((size_t)N*D); A.xhatf = zalloc((size_t)N*D); A.rstdf = zalloc(N); A.xf = zalloc((size_t)N*D);
    A.logits = zalloc((size_t)N*C.V);
    for (int l = 0; l < C.L; l++) {
        A.xhat1[l] = zalloc((size_t)N*D); A.rstd1[l] = zalloc(N); A.xn[l] = zalloc((size_t)N*D);
        A.Q[l] = zalloc((size_t)N*D); A.K[l] = zalloc((size_t)N*D); A.Vv[l] = zalloc((size_t)N*D);
        A.attn[l] = zalloc((size_t)C.B*H*T*T); A.ctx[l] = zalloc((size_t)N*D); A.x_mid[l] = zalloc((size_t)N*D);
        A.xhat2[l] = zalloc((size_t)N*D); A.rstd2[l] = zalloc(N); A.xn2[l] = zalloc((size_t)N*D);
        A.h[l] = zalloc((size_t)N*F); A.hact[l] = zalloc((size_t)N*F); A.x_out[l] = zalloc((size_t)N*D);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s check|bench|train refdir [reps]\n", argv[0]); return 1; }
    const char *mode = argv[1], *dir = argv[2];
    char cfgpath[512]; snprintf(cfgpath, sizeof cfgpath, "%s/model.cfg", dir);
    FILE *f = fopen(cfgpath, "r"); if (!f) { perror(cfgpath); return 1; }
    if (fscanf(f, "%d %d %d %d %d %d %d %d %lf", &C.V,&C.S,&C.D,&C.H,&C.F,&C.L,&C.B,&C.steps,&C.lr) != 9) { fprintf(stderr, "bad cfg\n"); return 1; }
    fclose(f);
    /* size the flat blocks */
    Params tmp; layout(&tmp, (double*)0, &C); size_t n = tmp.n;
    pbuf = zalloc(n); gbuf = zalloc(n); mbuf = zalloc(n); vbuf = zalloc(n);
    layout(&P, pbuf, &C); layout(&G, gbuf, &C);
    size_t bytes; double *p0 = slurp(dir, "params0.bin", &bytes);
    if (bytes != n*sizeof(double)) { fprintf(stderr, "params0.bin size mismatch: %zu vs %zu\n", bytes, n*sizeof(double)); return 1; }
    memcpy(pbuf, p0, bytes);
    int *batches = slurp(dir, "batches.bin", NULL);
    size_t per = (size_t)C.B*(C.S+1);
    alloc_acts();
    double *dl = zalloc((size_t)C.B*(C.S+1)*C.V);
    printf("OpenBLAS: %s | threads: %d | params: %zu\n", openblas_get_config(), openblas_get_num_threads(), n);

    if (!strcmp(mode, "check")) {
        forward(batches);
        double *ref_lg = slurp(dir, "logits0.bin", NULL), *ref_g = slurp(dir, "grads0.bin", NULL);
        double loss = loss_and_dlogits(batches, dl);
        backward(batches, dl);
        printf("max|logits - ref| = %.3e\n", maxdiff(A.logits, ref_lg, (size_t)C.B*(C.S+1)*C.V));
        printf("max|grads  - ref| = %.3e\n", maxdiff(gbuf, ref_g, n));
        printf("step-0 loss = %.12f\n", loss);
        int ok = maxdiff(A.logits, ref_lg, (size_t)C.B*(C.S+1)*C.V) < 1e-9 && maxdiff(gbuf, ref_g, n) < 1e-9;
        printf(ok ? "PASS\n" : "FAIL\n"); return ok ? 0 : 2;
    }
    if (!strcmp(mode, "train")) {
        double *ref = slurp(dir, "losses.bin", NULL); double worst = 0;
        for (int s = 0; s < C.steps; s++) {
            const int *ids = batches + (size_t)s*per;
            forward(ids); double loss = loss_and_dlogits(ids, dl); backward(ids, dl); adam_step();
            double d = fabs(loss - ref[s]); if (d > worst) worst = d;
            printf("step %3d  loss %.6f  numpy %.6f  diff %.2e\n", s+1, loss, ref[s], d);
        }
        printf("worst loss diff = %.3e  %s\n", worst, worst < 1e-8 ? "PASS" : "FAIL"); return worst < 1e-8 ? 0 : 2;
    }
    if (!strcmp(mode, "bench")) {
        int reps = argc > 3 ? atoi(argv[3]) : 20; double *t[5];
        for (int i = 0; i < 5; i++) t[i] = zalloc(reps);
#ifdef USE_ITT
        dom = __itt_domain_create("tx");
        sh_fwd = __itt_string_handle_create("forward");  sh_loss = __itt_string_handle_create("loss");
        sh_bwd = __itt_string_handle_create("backward"); sh_adam = __itt_string_handle_create("adam");
#endif
        for (int r = -3; r < reps; r++) {       /* 3 warm-up runs */
            double a = now(); ITT_BEGIN(sh_fwd);  forward(batches); ITT_END();
            double b = now(); ITT_BEGIN(sh_loss); double loss = loss_and_dlogits(batches, dl); (void)loss; ITT_END();
            double c = now(); ITT_BEGIN(sh_bwd);  backward(batches, dl); ITT_END();
            double d = now(); ITT_BEGIN(sh_adam); adam_step(); ITT_END();
            double e = now();
            if (r >= 0) { t[0][r]=b-a; t[1][r]=c-b; t[2][r]=d-c; t[3][r]=e-d; t[4][r]=e-a; }
        }
        const char *nm[5] = {"forward","loss","backward","adam","train step"};
        for (int i = 0; i < 5; i++) {
            for (int a = 0; a < reps; a++) for (int b = a+1; b < reps; b++) if (t[i][b] < t[i][a]) { double x = t[i][a]; t[i][a] = t[i][b]; t[i][b] = x; }
            double med = (reps % 2) ? t[i][reps/2] : 0.5*(t[i][reps/2-1] + t[i][reps/2]);
            printf("%-11s median %9.3f ms  min %9.3f ms  max %9.3f ms\n", nm[i], med*1e3, t[i][0]*1e3, t[i][reps-1]*1e3);
        }
        return 0;
    }
    return 1;
}
