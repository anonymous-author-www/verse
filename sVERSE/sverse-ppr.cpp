#include <omp.h>
#include <math.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

#define SIGMOID_BOUND 6

typedef unsigned long long ull;

bool silent = false;
int n_threads = 6;
float global_lr = 0.0025f;
int n_epochs = 100000;
int n_hidden = 128;
int n_samples = 3;
float ppralpha = 0.85f;

ull total_steps;
ull step = 0;

ull nv = 0, ne = 0;
int *offsets;
int *edges;
int* degrees;

float *w0;

const int sigmoid_table_size = 1024;
float *sigmoid_table;
const float SIGMOID_RESOLUTION = sigmoid_table_size / (SIGMOID_BOUND * 2.0f);

uint64_t rng_seed[2];

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// http://xoroshiro.di.unimi.it/#shootout
uint64_t lrand() {
    const uint64_t s0 = rng_seed[0];
    uint64_t s1 = rng_seed[1];
    const uint64_t result = s0 + s1;
    s1 ^= s0;
    rng_seed[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);  // a, b
    rng_seed[1] = rotl(s1, 36);                    // c
    return result;
}

static inline double drand() {
    const union un {
        uint64_t i;
        double d;
    } a = {UINT64_C(0x3FF) << 52 | lrand() >> 12};
    return a.d - 1.0;
}

void InitSigmoidTable() {
    float x;
    sigmoid_table = (float *)malloc((sigmoid_table_size + 1) * sizeof(float));
    for (int k = 0; k != sigmoid_table_size; k++) {
        x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
        sigmoid_table[k] = 1 / (1 + exp(-x));
    }
}

float FastSigmoid(float x) {
    if (x > SIGMOID_BOUND)
        return 1;
    else if (x < -SIGMOID_BOUND)
        return 0;
    int k = (x + SIGMOID_BOUND) * SIGMOID_RESOLUTION;
    return sigmoid_table[k];
}

inline int irand(int min,  int max) { return lrand() % (max - min) + min; }

inline int sample_neighbor(int node) {
    return edges[irand(offsets[node], offsets[node + 1])];
}

inline int sample_rw(int node) {
    int n2 = node;
    while (drand() < ppralpha) n2 = sample_neighbor(n2);
    return n2;
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++)
        if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    return -1;
}

inline void update(float* w_s, float* w_t, int label, const float bias) {
    float score = -bias;
#pragma simd
    for (int c = 0; c < n_hidden; c++) score += w_s[c] * w_t[c];
    score = (label - FastSigmoid(score)) * global_lr;
#pragma simd
    for (int c = 0; c < n_hidden; c++) w_t[c] += score * w_s[c];
#pragma simd
    for (int c = 0; c < n_hidden; c++) w_s[c] += score * w_t[c];
}

void Train() {
#pragma omp parallel num_threads(n_threads)
    {
        const float nce_bias = log(nv);
        const float nce_bias_neg = log(n_samples) + log(nv);
        int tid = omp_get_thread_num();
        ull last_ncount = 0;
        ull ncount = 0;
        float lr = global_lr;
#pragma omp barrier
        while (1) {
            if (ncount - last_ncount > 10000) {
                ull diff = ncount - last_ncount;
#pragma omp atomic
                step += diff;
                if (step > total_steps) break;
				if (tid == 0)
				  if (!silent)
					cout << fixed << setprecision(6) << "\rlr " << lr << ", Progress "
					<< std::setprecision(2) << step / (float)(total_steps + 1) * 100
					<< "%";
                last_ncount = ncount;
            }
            if (w0[0] != w0[0]) {
                printf("\nNaN loss! Aborting..");
                break;
            }
            lr = global_lr * (1 - step / (float)(total_steps + 1));
            lr = max(lr, 1e-5f);
            int n1 = irand(0, nv);
            int n2 = sample_rw(n1);
            update(&w0[n1 * n_hidden], &w0[n2 * n_hidden], 1, nce_bias);
            for (int i = 0; i < n_samples; i++) {
                int neg = irand(0, nv);
                update(&w0[n1 * n_hidden], &w0[neg * n_hidden], 0, nce_bias_neg);
            }
            ncount++;
        }
    }
}

int main(int argc, char **argv) {
    int a = 0;
    string network_file, embedding_file;
    ull x = time(NULL);
    for (int i = 0; i < 2; i++) {
        ull z = (x += UINT64_C(0x9E3779B97F4A7C15));
        z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
        rng_seed[i] = z ^ (z >> 31);
    }
    InitSigmoidTable();
    if ((a = ArgPos((char *)"-input", argc, argv)) > 0)
        network_file = argv[a + 1];
    else {
        cout << "Input file not given! Aborting now.." << endl;
        getchar();
        return 0;
    }
    if ((a = ArgPos((char *)"-output", argc, argv)) > 0)
        embedding_file = argv[a + 1];
    else {
        cout << "Output file not given! Aborting now.." << endl;
        getchar();
        return 0;
    }
    if ((a = ArgPos((char *)"-dim", argc, argv)) > 0)
        n_hidden = atoi(argv[a + 1]);
	if ((a = ArgPos((char *)"-silent", argc, argv)) > 0)
	  silent = true;
    if ((a = ArgPos((char *)"-nsamples", argc, argv)) > 0)
        n_samples = atoi(argv[a + 1]);
    if ((a = ArgPos((char *)"-threads", argc, argv)) > 0)
        n_threads = atoi(argv[a + 1]);
    if ((a = ArgPos((char *)"-steps", argc, argv)) > 0)
        n_epochs = atoi(argv[a + 1]);
    if ((a = ArgPos((char *)"-lr", argc, argv)) > 0)
        global_lr = atof(argv[a + 1]);
    if ((a = ArgPos((char *)"-alpha", argc, argv)) > 0)
        ppralpha = atof(argv[a + 1]);
    ifstream embFile(network_file, ios::in | ios::binary);
    if (embFile.is_open()) {
        char header[] = "----";
        embFile.seekg(0, ios::beg);
        embFile.read(header, 4);
        if (strcmp(header, "XGFS") != 0) {
            cout << "Invalid header!: " << header << endl;
            getchar();
            return 1;
        }
        embFile.read(reinterpret_cast<char *>(&nv), sizeof(long long));
        embFile.read(reinterpret_cast<char *>(&ne), sizeof(long long));
        offsets = (int *)malloc((nv + 1) * sizeof(int32_t));
        edges = (int *)malloc(ne * sizeof(int32_t));
        embFile.read(reinterpret_cast<char *>(offsets), nv * sizeof(int32_t));
        offsets[nv] = (int)ne;
        embFile.read(reinterpret_cast<char *>(edges), sizeof(int32_t) * ne);
        cout << "nv: " << nv << ", ne: " << ne << endl;
        embFile.close();
    } else {
        return 0;
    }
    w0 = (float *)malloc(nv * n_hidden * sizeof(float));
    for (int i = 0; i < nv * n_hidden; i++) w0[i] = drand() - 0.5;
	degrees = (int *)malloc(nv * sizeof(int));
	for (int i = 0; i < nv; i++) degrees[i] = offsets[i + 1] - offsets[i];
    total_steps = n_epochs * (long long)nv;
    cout << "Total steps (mil): " << total_steps / 1000000. << endl;
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    Train();
    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    cout << endl
        << "Calculations took "
        << std::chrono::duration_cast<std::chrono::duration<float>>(end - begin).count()
        << " s to run"
        << endl;
	if (w0[0] != w0[0]) {
	  printf("\nNaN! Not saving the result..");
	  return 1;
	}
    ofstream output(embedding_file, std::ios::binary);
    output.write(reinterpret_cast<char *>(w0), sizeof(float) * n_hidden * nv);
    output.close();
}
