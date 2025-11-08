#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

// ------------------------------
// 1D pointer-jumping over fR and fQ
// Dead-end = n  → self loop
// ------------------------------

// Build fR(i) = Q[R[i]] with dead-end handling
__global__ void build_fR(const int* R, const int* Q, int n, int* fR) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = R[i];
    if (r >= n) { fR[i] = i; return; }          // row dead end
    int q = Q[r];
    if (q >= n) { fR[i] = i; return; }          // col-side dead end
    fR[i] = q;                                   // next row
}

// Build fQ(j) = R[Q[j]] with dead-end handling
__global__ void build_fQ(const int* R, const int* Q, int n, int* fQ) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int q = Q[j];
    if (q >= n) { fQ[j] = j; return; }          // col dead end
    int r = R[q];
    if (r >= n) { fQ[j] = j; return; }          // row-side dead end
    fQ[j] = r;                                   // next col
}

// One doubling step: next[x] = next[next[x]]
__global__ void jump_once(int* next, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    int y = next[x];
    next[x] = next[y];
}

// Mark candidates whose representative equals rep (after convergence)
__global__ void mark_candidates_1d(const int* next, int rep, unsigned char* cand, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    cand[x] = (next[x] == rep) ? 1 : 0;
}

// Flag “has predecessor in candidates”: for each candidate u, mark v = f[u]
__global__ void flag_preds(const unsigned char* cand, const int* f, unsigned char* has_pred, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    if (!cand[u]) return;
    int v = f[u];
    // since v in [0..n-1], plain write suffices (no race on same byte value)
    has_pred[v] = 1;
}

// Final in_cycle = candidate & has_pred  (true cycle nodes in 1D map)
__global__ void finalize_cycle_1d(const unsigned char* cand, const unsigned char* has_pred,
                                  unsigned char* in_cycle, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    in_cycle[x] = (cand[x] && has_pred[x]) ? 1 : 0;
}

int main() {
    // -------------------- Example --------------------
    // EDIT HERE:
    // const int n = 4;
    // const int k = 0;   // start row
    // const int l = 1;   // start col
    // int h_R[n] = {0, 1, 4, 4}; // 4 == dead-end (n)
    // int h_Q[n] = {1, 0, 4, 4}; // 4 == dead-end (n)

    const int n = 8, k = 6, l = 1;
    int h_R[n] = {8,5,7,3,4,6,1,8};
    int h_Q[n] = {8,5,8,6,5,2,3,4};

    // const int n = 8, k = 7, l = 0;
    // int h_R[n] = {2, 8, 8, 1, 8, 3, 6, 0};
    // int h_Q[n] = {6, 7, 6, 3, 8, 8, 5, 8};
    // Expected pair cycle: (0,1)->(0,0)->(1,0)->(1,1)->(0,1)

    // -------------------- Device buffers --------------------
    int *d_R, *d_Q;
    int *d_fR, *d_fQ;                 // composed maps
    unsigned char *d_candR, *d_candQ;
    unsigned char *d_hasPredR, *d_hasPredQ;
    unsigned char *d_cycR, *d_cycQ;

    cudaMalloc(&d_R, n*sizeof(int));
    cudaMalloc(&d_Q, n*sizeof(int));
    cudaMalloc(&d_fR, n*sizeof(int));
    cudaMalloc(&d_fQ, n*sizeof(int));
    cudaMalloc(&d_candR, n*sizeof(unsigned char));
    cudaMalloc(&d_candQ, n*sizeof(unsigned char));
    cudaMalloc(&d_hasPredR, n*sizeof(unsigned char));
    cudaMalloc(&d_hasPredQ, n*sizeof(unsigned char));
    cudaMalloc(&d_cycR, n*sizeof(unsigned char));
    cudaMalloc(&d_cycQ, n*sizeof(unsigned char));

    cudaMemcpy(d_R, h_R, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, n*sizeof(int), cudaMemcpyHostToDevice);

    // -------------------- 1) Build composed maps --------------------
    dim3 block(128);
    dim3 grid((n + block.x - 1)/block.x);

    build_fR<<<grid, block>>>(d_R, d_Q, n, d_fR);
    build_fQ<<<grid, block>>>(d_R, d_Q, n, d_fQ);
    cudaDeviceSynchronize();

    // -------------------- 2) Pointer jumping on 1D maps --------------------
    int rounds = 0; for (int t=n; t>0; t >>= 1) ++rounds; // ceil(log2(n)) + 0
    for (int r=0; r<rounds; ++r) {
        jump_once<<<grid, block>>>(d_fR, n);
        jump_once<<<grid, block>>>(d_fQ, n);
        cudaDeviceSynchronize();
    }

    // Representatives (after convergence)
    int repR=0, repQ=0;
    cudaMemcpy(&repR, d_fR + k, sizeof(int), cudaMemcpyDeviceToHost); // fR[k]
    cudaMemcpy(&repQ, d_fQ + l, sizeof(int), cudaMemcpyDeviceToHost); // fQ[l]

    // -------------------- 3) Extract actual cycle nodes in 1D --------------------
    // (a) rows: candidates -> has_pred -> in_cycle
    mark_candidates_1d<<<grid, block>>>(d_fR, repR, d_candR, n);
    cudaMemset(d_hasPredR, 0, n*sizeof(unsigned char));
    flag_preds<<<grid, block>>>(d_candR, d_R /*NOT fR!*/, d_hasPredR, n);
    finalize_cycle_1d<<<grid, block>>>(d_candR, d_hasPredR, d_cycR, n);

    // (b) cols: candidates -> has_pred -> in_cycle
    mark_candidates_1d<<<grid, block>>>(d_fQ, repQ, d_candQ, n);
    cudaMemset(d_hasPredQ, 0, n*sizeof(unsigned char));
    flag_preds<<<grid, block>>>(d_candQ, d_Q /*NOT fQ!*/, d_hasPredQ, n);
    finalize_cycle_1d<<<grid, block>>>(d_candQ, d_hasPredQ, d_cycQ, n);
    cudaDeviceSynchronize();

    // -------------------- 4) Host: reconstruct the pair cycle from (k,l) --------------------
    // We have the row-cycle set and col-cycle set. Walk alternation from (k,l) to list the pair cycle.
    std::vector<unsigned char> h_cycR(n), h_cycQ(n);
    cudaMemcpy(h_cycR.data(), d_cycR, n*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cycQ.data(), d_cycQ, n*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Alternate (CPU) until we return to (k,l). Dead-ends are impossible inside cycles by construction.
    std::vector<std::pair<int,int>> pair_cycle;
    int i = k, j = l;
    do {
        pair_cycle.emplace_back(i,j);
        int j_next = (h_R[i] < n) ? h_R[i] : j;      // R[i]
        int i_next = (h_Q[j] < n) ? h_Q[j] : i;      // Q[j]
        i = i_next; j = j_next;
    } while (!(i == k && j == l) && (int)pair_cycle.size() <= 2*n);

    // -------------------- Print results --------------------
    printf("repR(fR[k=%d]) = %d, repQ(fQ[l=%d]) = %d\n", k, repR, l, repQ);

    printf("row-cycle indices: ");
    for (int x=0; x<n; ++x) if (h_cycR[x]) printf("%d ", x);
    printf("\n");

    printf("col-cycle indices: ");
    for (int x=0; x<n; ++x) if (h_cycQ[x]) printf("%d ", x);
    printf("\n");

    printf("pair cycle from (k=%d,l=%d): ", k, l);
    for (auto &p : pair_cycle) printf("(%d,%d) ", p.first, p.second);
    printf("\n");

    // -------------------- Cleanup --------------------
    cudaFree(d_R); cudaFree(d_Q);
    cudaFree(d_fR); cudaFree(d_fQ);
    cudaFree(d_candR); cudaFree(d_candQ);
    cudaFree(d_hasPredR); cudaFree(d_hasPredQ);
    cudaFree(d_cycR); cudaFree(d_cycQ);
    return 0;
}
