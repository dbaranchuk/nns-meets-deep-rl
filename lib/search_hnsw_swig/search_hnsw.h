#include <vector>
#include <cstdio>
#include <queue>
#include <iostream>
#include <assert.h>
#include <omp.h>
#include <cmath>
#include <random>
#include <time.h>
#include <unordered_set>
#include <stdlib.h>


#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else
#include <x86intrin.h>
#endif

#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif


typedef unsigned idx_t;

float fvec_L2sqr(const float *x, const float *y, size_t d);

void find_nearest(int nb, int d1, float *vertices,               // matrix [n_vertices, vec_dimension]
                  int nb1, int max_degree, int *edges,           // matrix [n_vertices, max_degree]
                  int nb2, int max_degree1, float *edge_probs,   // matrix [n_vertices, max_degree]
                  int nq, int d, float *queries,                 // matrix [n_queries, vec_dimension]
                  int nq1, int max_path,  int *trajectories,     // matrix [n_queries, max_path]
                  int nq3, int num_actions, float *samples,      // matrix [n_queries, num_actions] num_actions = max_degree * max_path
                  int nq2, int num_results, int *results,        // matrix [n_queries, 3 + num_actions]
                  int *k,                                        // number
                  int *initial_vertex_id,                        // number
                  int *ef,                                       // number
                  int *nt);                                      // number