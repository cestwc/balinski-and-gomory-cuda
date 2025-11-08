#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define IDX2C(i,j,n) ((j)*(n)+(i))

// ===============================================================
// 1) Build composed maps with dead-end handling (UNCHANGED)
// ===============================================================
__global__ void build_fR(const int* R, const int* Q, int n, int* fR) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = R[i];
    if (r >= n) { fR[i] = i; return; }
    int q = Q[r];
    if (q >= n) { fR[i] = i; return; }
    fR[i] = q;
}
__global__ void build_fQ(const int* R, const int* Q, int n, int* fQ) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int q = Q[j];
    if (q >= n) { fQ[j] = j; return; }
    int r = R[q];
    if (r >= n) { fQ[j] = j; return; }
    fQ[j] = r;
}

// pointer jumping once (UNCHANGED)
__global__ void jump_once(int* next, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    int y = next[x];
    next[x] = next[y];
}

// mark candidates / finalize 1D cycles (UNCHANGED)
__global__ void mark_candidates_1d(const int* next, int rep, unsigned char* cand, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    cand[x] = (next[x] == rep) ? 1 : 0;
}
__global__ void flag_preds(const unsigned char* cand, const int* f, unsigned char* has_pred, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    if (!cand[u]) return;
    int v = f[u];
    has_pred[v] = 1;
}
__global__ void finalize_cycle_1d(const unsigned char* cand,
                                  const unsigned char* has_pred,
                                  unsigned char* in_cycle, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    in_cycle[x] = (cand[x] && has_pred[x]) ? 1 : 0;
}

// ===============================================================
// 2) Flip exactly the specific (i,j) pairs in the cycle (FIXED)
// ===============================================================
__global__ void flip_specific_pairs(int* d_X,
                                    const int* cyc_is,
                                    const int* cyc_js,
                                    int m, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m) return;
    int i = cyc_is[t];
    int j = cyc_js[t];
    int idx = IDX2C(i,j,n);
    d_X[idx] = 1 - d_X[idx];
}

int main() {
    // ---------- Example ----------
    const int n = 4, k = 0, l = 1;
    int h_R[n] = {0, 1, 4, 4}; // 4 == dead-end(n)
    int h_Q[n] = {1, 0, 4, 4};

    // An example X you want to flip on the cycle
    int h_X[n*n] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };

    // ---------- Device buffers ----------
    int *d_R, *d_Q, *d_fR, *d_fQ, *d_X;
    unsigned char *d_candR, *d_hasPredR, *d_cycR;
    unsigned char *d_candQ, *d_hasPredQ, *d_cycQ;

    cudaMalloc(&d_R, n*sizeof(int));
    cudaMalloc(&d_Q, n*sizeof(int));
    cudaMalloc(&d_fR, n*sizeof(int));
    cudaMalloc(&d_fQ, n*sizeof(int));
    cudaMalloc(&d_X, n*n*sizeof(int));
    cudaMalloc(&d_candR, n);
    cudaMalloc(&d_hasPredR, n);
    cudaMalloc(&d_cycR, n);
    cudaMalloc(&d_candQ, n);
    cudaMalloc(&d_hasPredQ, n);
    cudaMalloc(&d_cycQ, n);

    cudaMemcpy(d_R, h_R, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, n*n*sizeof(int), cudaMemcpyHostToDevice);

    // ---------- 1) Cycle detection (UNCHANGED) ----------
    dim3 block1(128);
    dim3 grid1((n + block1.x - 1)/block1.x);

    build_fR<<<grid1, block1>>>(d_R, d_Q, n, d_fR);
    build_fQ<<<grid1, block1>>>(d_R, d_Q, n, d_fQ);
    cudaDeviceSynchronize();

    int rounds = 0; for (int t=n; t>0; t >>= 1) ++rounds;
    for (int r=0; r<rounds; ++r) {
        jump_once<<<grid1, block1>>>(d_fR, n);
        jump_once<<<grid1, block1>>>(d_fQ, n);
    }
    cudaDeviceSynchronize();

    int repR, repQ;
    cudaMemcpy(&repR, d_fR + k, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&repQ, d_fQ + l, sizeof(int), cudaMemcpyDeviceToHost);

    mark_candidates_1d<<<grid1, block1>>>(d_fR, repR, d_candR, n);
    cudaMemset(d_hasPredR, 0, n);
    flag_preds<<<grid1, block1>>>(d_candR, d_fR, d_hasPredR, n); // use fR
    finalize_cycle_1d<<<grid1, block1>>>(d_candR, d_hasPredR, d_cycR, n);

    mark_candidates_1d<<<grid1, block1>>>(d_fQ, repQ, d_candQ, n);
    cudaMemset(d_hasPredQ, 0, n);
    flag_preds<<<grid1, block1>>>(d_candQ, d_fQ, d_hasPredQ, n); // use fQ
    finalize_cycle_1d<<<grid1, block1>>>(d_candQ, d_hasPredQ, d_cycQ, n);
    cudaDeviceSynchronize();

    // ---------- Copy in-cycle masks to host (for printing & pair reconstruction) ----------
    std::vector<unsigned char> h_cycR(n), h_cycQ(n);
    cudaMemcpy(h_cycR.data(), d_cycR, n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cycQ.data(), d_cycQ, n, cudaMemcpyDeviceToHost);

    // ---------- Reconstruct the *pair* cycle from (k,l) on host ----------
    std::vector<std::pair<int,int>> pair_cycle;
    int i = k, j = l;
    do {
        pair_cycle.emplace_back(i,j);
        int j_next = (h_R[i] < n) ? h_R[i] : j; // R[i]
        int i_next = (h_Q[j] < n) ? h_Q[j] : i; // Q[j]
        i = i_next; j = j_next;
    } while (!(i == k && j == l) && (int)pair_cycle.size() <= 2*n);

    // Print cycle detection info
    printf("repR(fR[k=%d]) = %d, repQ(fQ[l=%d]) = %d\n", k, repR, l, repQ);
    printf("row-cycle indices: "); for (int x=0; x<n; ++x) if (h_cycR[x]) printf("%d ", x); printf("\n");
    printf("col-cycle indices: "); for (int x=0; x<n; ++x) if (h_cycQ[x]) printf("%d ", x); printf("\n");

    printf("pair cycle from (k=%d,l=%d): ", k, l);
    for (auto &p : pair_cycle) printf("(%d,%d) ", p.first, p.second);
    printf("\n");

    // ---------- FIXED: flip ONLY those pair-cycle entries ----------
    const int m = (int)pair_cycle.size();
    std::vector<int> h_cyc_is(m), h_cyc_js(m);
    for (int t=0; t<m; ++t) { h_cyc_is[t] = pair_cycle[t].first; h_cyc_js[t] = pair_cycle[t].second; }

    int *d_cyc_is = nullptr, *d_cyc_js = nullptr;
    cudaMalloc(&d_cyc_is, m*sizeof(int));
    cudaMalloc(&d_cyc_js, m*sizeof(int));
    cudaMemcpy(d_cyc_is, h_cyc_is.data(), m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cyc_js, h_cyc_js.data(), m*sizeof(int), cudaMemcpyHostToDevice);

    dim3 block2(128);
    dim3 grid2((m + block2.x - 1)/block2.x);
    flip_specific_pairs<<<grid2, block2>>>(d_X, d_cyc_is, d_cyc_js, m, n);
    cudaDeviceSynchronize();

    // Copy back X and print which entries were flipped
    cudaMemcpy(h_X, d_X, n*n*sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nFlipped entries (i,j): ");
    for (int t=0; t<m; ++t) printf("(%d,%d) ", h_cyc_is[t], h_cyc_js[t]);
    printf("\n");

    printf("\nFlipped d_X (column-major):\n");
    for (int ii=0; ii<n; ++ii) {
        for (int jj=0; jj<n; ++jj)
            printf("%2d ", h_X[IDX2C(ii,jj,n)]);
        printf("\n");
    }

    // cleanup
    cudaFree(d_R); cudaFree(d_Q); cudaFree(d_fR); cudaFree(d_fQ); cudaFree(d_X);
    cudaFree(d_candR); cudaFree(d_hasPredR); cudaFree(d_cycR);
    cudaFree(d_candQ); cudaFree(d_hasPredQ); cudaFree(d_cycQ);
    cudaFree(d_cyc_is); cudaFree(d_cyc_js);
    return 0;
}
