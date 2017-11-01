#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include "mkl.h"

#include <omp.h>

// These three parameters control the neural network operation, and were used to
// test different models for producing sim_e

// Embedding lookup is used to compute the hidden layer values (and the
// gradients) for the case when input vector x is one-hot. 1 used for the paper
#define USE_EMBEDDING_LOOKUP 1

// Defines tanh nonlinearity before softmax. Adds ~0.2% performance for the
// classification and similarity reconstruction when using cosine similarity,
// adding some computational cost and model complication. 0 is used in the paper
#define USE_TANH_NONLINEARITY 0

// Defines the node importances according to PageRank of the nodes. This is one
// of the things that did not go to the paper. Significantly improves the
// performance in the classification, but lowers it at the reconstruction task.
// 0 is used in the paper
#define USE_PR_WEIGHTS 0

// Default parameters for sampling and ppr
bool use_sampling = 0;
bool precompute_ppr = 1;

// Controls the verbosity of the output. Useful for huge graphs.
bool verbose = 0;

// Some additional parameter declarations for different optimizers
typedef struct { float* v; } MomentumParams; // Momentum uses 2*W memory
typedef struct { float* v; } NAGParams; // NAG uses 2*W memory
typedef struct {
  float* v;
  float* m;
} AdamParams;  // Adam uses 3*W memory

#define MAX_STRING 256
#define min(x, y) (x > y ? y : x)
#define max(x, y) (x > y ? x : y)

// Initialized the random number table. As we do not need any true randomness, this can be done for efficiency.
void InitRandomTable(VSLStreamStatePtr* rng_stream);
float FastRandom();
int FastRandrange(int min, int max);

// For cross-platform argument parsing. boost:program_options is a mess.
int ArgPos(char* str, int argc, char** argv);

// Shuffles the array
void Shuffle(int* a, int N);

// Main train thread
void Train();

// Various similarities with a unified interface
void estimate_pr_rw(float* prs);
void estimate_ppr_rw(int seednode, float* pprs);
void estimate_ppr_poweriter_csr(int seednode, float* pprs);
void estimate_pr_poweriter_csr(float* prs);
void jaccard(int seednode, float* sim);
void adamicadar(int i, float* sim);
void neigh(int i, float* sim);
void cosine(int i, float* sim);

// Helper functions for similarity calculation
int degree(int node);
void common_neighbors(int i, int j, int* res_or, int* res_and);

void simrank(float c); // Please, do not use

// Various optimizers with a unified interface
void SGD(int sz, float lr, float* w, float* dw, void* params);
void Momentum(int sz, float lr, float* w, float* dw, void* params);
void NAG(int sz, float lr, float* w, float* dw, void* params);
void Adam(int sz, float lr, float* w, float* dw, void* params);
float Adam_lr(float lr);

const int random_table_size = 1024 * 1024 * 1; // 4 Mb

void (*similarity)(int, float*) = &estimate_ppr_rw;
void (*optimizer)(int, float, float*, float*, void*) = &Adam;
float global_lr = 0.005;
int n_epochs = 250;
int n_hidden = 128, n_batch = 128;
int n_samples = 3000;
MKL_INT nv = 0, ne = 0;

int* r_offsets;
int* r_edges;
float* prs;

MKL_INT* offsets;
MKL_INT* edges;

float* elems;

float sr_diff;
float sr_c = 0.95;
float* similarity_table;
float* partials_table; //For more efficient SimRank computation
int ppr_step = 0;

float* w0;

// Optimizers parameters
const float momentum_1 = 0.9, momentum_2 = 0.999, epsilon = 1e-8;

// PPR parameters
float alpha = 0.80;
const float tol = 1e-6;
const int maxiter = 999;

char network_file[MAX_STRING], embedding_file[MAX_STRING],
    optimizer_name[MAX_STRING], similarity_name[MAX_STRING];
int is_binary = 0, num_threads = 1;
int step, total_steps;

VSLStreamStatePtr rng_stream;
float* random_table;
unsigned long long nextfloatidx = 0;