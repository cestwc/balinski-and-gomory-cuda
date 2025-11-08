#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define IDX2C(i,j,n) ((j)*(n)+(i))

// =========================================================
// 1. Build composed mappings fR(i)=Q[R[i]], fQ(j)=R[Q[j]]
// =========================================================
__global__ void build_composed_maps(const int* R, const int* Q,
                                    int n, int* fR, int* fQ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) {
        // --- row side ---
        int r = R[x];
        if (r >= n) fR[x] = x;
        else {
            int q = Q[r];
            fR[x] = (q >= n ? x : q);
        }

        // --- col side ---
        int q = Q[x];
        if (q >= n) fQ[x] = x;
        else {
            int r2 = R[q];
            fQ[x] = (r2 >= n ? x : r2);
        }
    }
}

// =========================================================
// 2. Pointer jumping (path doubling)
// =========================================================
__global__ void jump_once(int* next, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    int y = next[x];
    next[x] = next[y];
}

// =========================================================
// 3. Fused cycle detection kernel
//    - replaces mark_candidates + flag_preds + finalize
// =========================================================
__global__ void detect_cycles_fused(const int* fR, const int* fQ,
                                    const int* R, const int* Q,
                                    int repR, int repQ,
                                    unsigned char* cycR,
                                    unsigned char* cycQ,
                                    int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;

    // ---- Row side ----
    bool candR = (fR[x] == repR);
    bool hasPredR = false;
    if (candR) {
        // If some other node in same set points to x
        int pred_idx = -1;
        // reverse check: does any i map to x under fR?
        // (cheap because we expect small n; for larger n we'd precompute)
        // We'll just use R[x] mapping logic since it's functional
        // This can be simplified: true cycle node always has fR[fR[x]]==fR[x]
        hasPredR = (fR[fR[x]] == repR);
    }
    cycR[x] = (candR && hasPredR);

    // ---- Column side ----
    bool candQ = (fQ[x] == repQ);
    bool hasPredQ = false;
    if (candQ) {
        hasPredQ = (fQ[fQ[x]] == repQ);
    }
    cycQ[x] = (candQ && hasPredQ);
}

// =========================================================
// 4. Flip kernel — flip X[i,j] for each (i,j) in the pair cycle
// =========================================================
__global__ void flip_specific_pairs(int* d_X,
                                    const int* cyc_is,
                                    const int* cyc_js,
                                    int m, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m) return;
    int i = cyc_is[t];
    int j = cyc_js[t];
    int idx = IDX2C(i, j, n);
    d_X[idx] = 1 - d_X[idx];
}

// =========================================================
// 5. Host-side driver
// =========================================================
int main()
{
    const int n = 4, k = 0, l = 1;
    int h_R[n] = {0, 1, 4, 4};
    int h_Q[n] = {1, 0, 4, 4};

    int h_X[n*n] = {
         1,0,0,0,
         0,1,0,0,
         0,0,1,0,
         0,0,0,1
    };
    float h_B[n*n] = {
        0,-2,-4,-3,
        0,0,-5,-5,
        2,4,0,-3,
        6,2,3,0
    };
    float h_V[n] = {2,5,5,4};

    // Allocate device buffers
    int *d_R, *d_Q, *d_fR, *d_fQ, *d_X;
    unsigned char *d_cycR, *d_cycQ;

    cudaMalloc(&d_R, n*sizeof(int));
    cudaMalloc(&d_Q, n*sizeof(int));
    cudaMalloc(&d_fR, n*sizeof(int));
    cudaMalloc(&d_fQ, n*sizeof(int));
    cudaMalloc(&d_cycR, n*sizeof(unsigned char));
    cudaMalloc(&d_cycQ, n*sizeof(unsigned char));
    cudaMalloc(&d_X, n*n*sizeof(int));

    cudaMemcpy(d_R, h_R, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, n*n*sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid((n + block.x - 1)/block.x);

    // 1. Build composed maps
    build_composed_maps<<<grid, block>>>(d_R, d_Q, n, d_fR, d_fQ);
    cudaDeviceSynchronize();

    // 2. Pointer jumping
    int rounds = 0; for (int t = n; t > 0; t >>= 1) ++rounds;
    for (int r = 0; r < rounds; ++r) {
        jump_once<<<grid, block>>>(d_fR, n);
        jump_once<<<grid, block>>>(d_fQ, n);
        cudaDeviceSynchronize();
    }

    // 3. Representatives
    int repR, repQ;
    cudaMemcpy(&repR, d_fR + k, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&repQ, d_fQ + l, sizeof(int), cudaMemcpyDeviceToHost);

    // 4. Detect cycles in one fused kernel
    detect_cycles_fused<<<grid, block>>>(d_fR, d_fQ, d_R, d_Q,
                                         repR, repQ, d_cycR, d_cycQ, n);
    cudaDeviceSynchronize();

    // Copy cycle results back
    std::vector<unsigned char> h_cycR(n), h_cycQ(n);
    cudaMemcpy(h_cycR.data(), d_cycR, n*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cycQ.data(), d_cycQ, n*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    printf("repR(fR[k=%d]) = %d, repQ(fQ[l=%d]) = %d\n", k, repR, l, repQ);

    printf("row-cycle indices: ");
    for (int i = 0; i < n; ++i)
        if (h_cycR[i]) printf("%d ", i);
    printf("\n");

    printf("col-cycle indices: ");
    for (int j = 0; j < n; ++j)
        if (h_cycQ[j]) printf("%d ", j);
    printf("\n");

    // 5. Reconstruct pair cycle on host
    std::vector<std::pair<int,int>> pair_cycle;
    int i = k, j = l;
    do {
        pair_cycle.emplace_back(i,j);
        int j_next = (h_R[i] < n) ? h_R[i] : j;
        int i_next = (h_Q[j] < n) ? h_Q[j] : i;
        i = i_next; j = j_next;
    } while (!(i == k && j == l) && pair_cycle.size() <= 2*n);

    printf("pair cycle from (k=%d,l=%d): ", k, l);
    for (auto &p : pair_cycle) printf("(%d,%d) ", p.first, p.second);
    printf("\n");

    // 6. Flip in-cycle pairs on GPU
    int m = pair_cycle.size();
    std::vector<int> h_cyc_is(m), h_cyc_js(m);
    for (int t = 0; t < m; ++t) {
        h_cyc_is[t] = pair_cycle[t].first;
        h_cyc_js[t] = pair_cycle[t].second;
    }
    int *d_cyc_is, *d_cyc_js;
    cudaMalloc(&d_cyc_is, m*sizeof(int));
    cudaMalloc(&d_cyc_js, m*sizeof(int));
    cudaMemcpy(d_cyc_is, h_cyc_is.data(), m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cyc_js, h_cyc_js.data(), m*sizeof(int), cudaMemcpyHostToDevice);

    flip_specific_pairs<<<(m+127)/128, 128>>>(d_X, d_cyc_is, d_cyc_js, m, n);
    cudaDeviceSynchronize();

    // 7. Copy back and print X
    cudaMemcpy(h_X, d_X, n*n*sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nFlipped d_X (column-major):\n");
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col)
            printf(" %d ", h_X[IDX2C(row,col,n)]);
        printf("\n");
    }

    // Cleanup
    cudaFree(d_R); cudaFree(d_Q);
    cudaFree(d_fR); cudaFree(d_fQ);
    cudaFree(d_cycR); cudaFree(d_cycQ);
    cudaFree(d_X);
    cudaFree(d_cyc_is); cudaFree(d_cyc_js);
    return 0;
}
