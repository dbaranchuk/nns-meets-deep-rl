#include "search_hnsw.h"

/*
* HNSW beam search algorithm
*/

void find_nearest(int nb, int d1, float *vertices,               // matrix [n_vertices, vec_dimension]
                  int nb1, int max_degree, int *edges,           // matrix [n_vertices, max_degree]
                  int nb2, int max_degree1, float *edge_probs,   // matrix [n_vertices, max_degree]
                  int nq, int d, float *queries,                 // matrix [n_queries, vec_dimension]
                  int nq1, int max_path, int *trajectories,      // matrix [n_queries, max_path]
                  int nq3, int num_actions, float *samples,      // matrix [n_queries, num_actions]
                  int nq2, int num_results, int *results,        // matrix [n_queries, 3 + num_actions]
                  int *k,                                        // number
                  int *initial_vertex_id,                        // number
                  int *ef,                                       // number
                  int *nt)                                       // number
                  {
    assert(nq == nq1 == nq2 == nq3);
    assert(d == d1);
    assert(nb == nb1 == nb2);
    assert(max_degree ==  max_degree1);
    assert(*nt > 0 && *ef > 0);
    assert(*k <= *ef);

#pragma omp parallel for num_threads(*nt)
    for (int32_t q = 0; q < nq; q++) {
        std::unordered_set <idx_t> visited_ids;
        std::priority_queue <std::pair<float, idx_t >> ef_top;
        std::priority_queue <std::pair<float, idx_t >> candidates;  // contains available edges

        const float *query = queries + d * q;
        const float *uniform_samples = samples + q * num_actions;

        int *actions = results + num_results * q + *k + 2;
        int *trajectory = trajectories + max_path * q;

        float distance;
        if (d == 300)  // Hard code for GloVe data
            distance = fvec_negative_dot(query, vertices + d * *initial_vertex_id, d);
        else
            distance = fvec_L2sqr(query, vertices + d * *initial_vertex_id, d);

        size_t num_dcs = 1;
        size_t num_hops = 0;

        ef_top.emplace(distance, *initial_vertex_id);
        candidates.emplace(-distance, *initial_vertex_id);
        visited_ids.insert(*initial_vertex_id);
        float lowerBound = distance;

        while (!candidates.empty()) {
            idx_t vertex_id = candidates.top().second;
            if (-candidates.top().first > lowerBound)
                break;

            candidates.pop();
            const int *neighbor_ids = edges + vertex_id * max_degree;
            const float *probs = edge_probs + vertex_id * max_degree;

            size_t j = 0;
            while (neighbor_ids[j] != -1 && j < (size_t) max_degree) {
                const idx_t neighbor_id = neighbor_ids[j];
                const float prob = probs[j];
                const float sample = uniform_samples[num_hops * max_degree + j];
                int *action = actions + num_hops * max_degree + j++;

                if (visited_ids.count(neighbor_id) > 0){
                    *action = -2;
                    continue;
                }
                if (prob < 0){
                    *action = -2;
                    continue;
                }
                *action = prob > sample;
                if (*action == 0) continue;

                visited_ids.insert(neighbor_id);
                if (d == 300)  // Hard code for GloVe data
                    distance = fvec_negative_dot(query, vertices + d * neighbor_id, d);
                else
                    distance = fvec_L2sqr(query, vertices + d * neighbor_id, d);
                num_dcs++;

                if (ef_top.top().first > distance || ef_top.size() < (size_t) *ef) {
                    candidates.emplace(-distance, neighbor_id);
                    ef_top.emplace(distance, neighbor_id);

                    if (ef_top.size() > (size_t) *ef)
                        ef_top.pop();
                    lowerBound = ef_top.top().first;
                }
                if (prob > 1)
                    *action = -2;
            }
            trajectory[num_hops++] = vertex_id;
            if (num_hops >= (size_t) max_path) break;
        }

        size_t answer_idx = *k - 1;
        while (!ef_top.empty()){
            if (ef_top.size() <= (size_t) *k)
                results[num_results * q + answer_idx--] = ef_top.top().second;
            ef_top.pop();
        }
        results[num_results * q + *k] = num_dcs;
        results[num_results * q + *k + 1] = num_hops;
    }
}

/** Fast L2 Distance Computation
 *
 * @param x
 * @param y
 * @param d - vector dimensionality
 * @return - l2 distance between x and y
*/
float fvec_L2sqr(const float *x, const float *y, size_t d) {
    float PORTABLE_ALIGN32 TmpRes[8];
#ifdef USE_AVX
    size_t qty16 = d >> 4;

        const float *pEnd1 = x + (qty16 << 4);

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (x < pEnd1) {
            v1 = _mm256_loadu_ps(x);
            x += 8;
            v2 = _mm256_loadu_ps(y);
            y += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(x);
            x += 8;
            v2 = _mm256_loadu_ps(y);
            y += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

        return (res);
#else
    size_t qty16 = d >> 4;

    const float *pEnd1 = x + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (x < pEnd1) {
        v1 = _mm_loadu_ps(x);
        x += 4;
        v2 = _mm_loadu_ps(y);
        y += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(x);
        x += 4;
        v2 = _mm_loadu_ps(y);
        y += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(x);
        x += 4;
        v2 = _mm_loadu_ps(y);
        y += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(x);
        x += 4;
        v2 = _mm_loadu_ps(y);
        y += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return (res);
#endif
}

/** Fast negative dot pruduct Computation (Needed for correct distance computations on GloVe1M)
 *
 * @param x
 * @param y
 * @param d - vector dimensionality
 * @return - l2 distance between x and y
*/
// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read (int d, const float *x)
{
    assert (0 <= d && d < 4);
    __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
    switch (d) {
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_load_ps (buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}

float fvec_negative_dot(const float *x, const float *y, size_t d)
{
    __m256 msum1 = _mm256_setzero_ps();

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps (x); x += 8;
        __m256 my = _mm256_loadu_ps (y); y += 8;
        msum1 = _mm256_add_ps (msum1, _mm256_mul_ps (mx, my));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 +=       _mm256_extractf128_ps(msum1, 0);

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        msum2 = _mm_add_ps (msum2, _mm_mul_ps (mx, my));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        msum2 = _mm_add_ps (msum2, _mm_mul_ps (mx, my));
    }

    msum2 = _mm_hadd_ps (msum2, msum2);
    msum2 = _mm_hadd_ps (msum2, msum2);
    return  -_mm_cvtss_f32 (msum2);
}
