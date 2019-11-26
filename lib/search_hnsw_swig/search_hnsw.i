%module search_hnsw

%{
    #define SWIG_FILE_WITH_INIT
    #include "search_hnsw.h"
%}

%include "numpy.i"
%include "typemaps.i"

%init %{
    import_array();
%}

%apply (int DIM1, int DIM2, float *IN_ARRAY2) {(int nb, int d1, float *vertices)}
%apply (int DIM1, int DIM2, int *IN_ARRAY2) {(int nb1, int max_degree, int *edges)}
%apply (int DIM1, int DIM2, float *IN_ARRAY2) {(int nb2, int max_degree1, float *edge_probs)}

%apply (int DIM1, int DIM2, float *IN_ARRAY2) {(int nq, int d, float *queries)}
%apply (int DIM1, int DIM2, int *INPLACE_ARRAY2) {(int nq1, int max_path, int *trajectories)}
%apply (int DIM1, int DIM2, float *IN_ARRAY2) {(int nq3, int num_actions, float *samples)}
%apply (int DIM1, int DIM2, int *INPLACE_ARRAY2) {(int nq2, int num_results, int *results)}

%apply int *INPUT {int *k, int *initial_vertex_id, int *ef, int *nt}

%include "search_hnsw.h"
