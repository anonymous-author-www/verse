#include "verse.h"

void InitRandomTable(VSLStreamStatePtr* rng_stream) {
  random_table = (float*)mkl_malloc(random_table_size * sizeof(float), 64);
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, *rng_stream, random_table_size,
               random_table, 0, 1);
}

// Returns a random number from the table from [0; 1]
float FastRandom() { return random_table[nextfloatidx++ % random_table_size]; }

// Returns a random number from the table from [min; max)
int FastRandrange(int min, int max) {
  return (int)((max - min) * FastRandom() + min);
}

// Estimates PPR using n_samples random walks from node 'seed'
void estimate_ppr_rw(int seed, float* pprs) {
  memset(pprs, 0, nv * sizeof(float));
  for (int i = 0; i < n_samples; i++) {
    int current = seed;
    while (FastRandom() < alpha)
      current = edges[FastRandrange(offsets[current], offsets[current + 1])];
    pprs[current] += 1. / n_samples;
  }
  float sum = cblas_sasum(
      nv, pprs, 1);  // Note that (1-a) is included in random walk sampling
  cblas_sscal(nv, 1 / sum, pprs, 1);
}

// Estimates PR using n_samples random walks
void estimate_pr_rw(float* prs) {
  memset(prs, 0, nv * sizeof(float));
  for (int i = 0; i < n_samples; i++) {
    int current = FastRandrange(0, nv);
    while (FastRandom() < alpha)
      current = edges[FastRandrange(offsets[current], offsets[current + 1])];
    prs[current] += 1. / n_samples;
  }
  float sum = cblas_sasum(
      nv, prs, 1);  // Note that (1-a) is included in random walk sampling
  cblas_sscal(nv, 1 / sum, prs, 1);
}

// Estimates PPR to a precision given by tol with power iteration
void estimate_ppr_poweriter_csr(int seednode, float* pprs) {
  float* temp = (float*)mkl_malloc(nv * sizeof(float), 64);
  float diff;

  for (int i = 0; i < nv; i++) pprs[i] = 1.f / nv;
  int curiter = 0;
  do {
    mkl_cspblas_scsrgemv("t", &nv, elems, offsets, edges, pprs, temp);
    cblas_sscal(nv, alpha, temp, 1);
    temp[seednode] += 1 - alpha;
    vsSub(nv, pprs, temp, pprs);
    diff = fabs(pprs[cblas_isamax(nv, pprs, 1)]);
    cblas_scopy(nv, temp, 1, pprs, 1);
  } while (curiter++ <= maxiter && diff > tol);
  mkl_free(temp);
}

// Estimates PR to a precision given by tol with power iteration
void estimate_pr_poweriter_csr(float* prs) {
  float* temp = (float*)mkl_malloc(nv * sizeof(float), 64);
  float diff;
  const float pradd = (1.f - alpha) / nv;
  for (int i = 0; i < nv; i++) prs[i] = 1. / nv;
  int curiter = 0;
  do {
    mkl_cspblas_scsrgemv("t", &nv, elems, offsets, edges, prs, temp);
    cblas_sscal(nv, alpha, temp, 1);
    cblas_saxpy(nv, 1., &pradd, 0, temp, 1);
    vsSub(nv, prs, temp, prs);
    diff = fabs(prs[cblas_isamax(nv, prs, 1)]);
    cblas_scopy(nv, temp, 1, prs, 1);
  } while (curiter++ <= maxiter && diff > tol);
  mkl_free(temp);
}

// Helper function. Counts neighbors of i and j and their intersection in an
// optimized single pass
void common_neighbors(int i, int j, int* res_or, int* res_and) {
  *res_or = 0;
  *res_and = 0;
  int i_s = offsets[i];
  int i_e = offsets[i + 1];
  int j_s = offsets[j];
  int j_e = offsets[j + 1];
  while (i_s < i_e || j_s < j_e) {
    int i_c = edges[i_s];
    int j_c = edges[j_s];
    if (i_c > j_c) {
      (*res_or)++;
      if (j_s < j_e)
        j_s++;
      else
        i_s++;
    } else if (i_c < j_c) {
      (*res_or)++;
      if (i_s < i_e)
        i_s++;
      else
        j_s++;
    } else {
      (*res_and)++;
      (*res_or)++;
      if (j_s < j_e) j_s++;
      if (i_s < i_e) i_s++;
    }
  }
}

// DO NOT USE. Computes SimRank in a dumbest possible way.
void simrank(float c) {
  for (int i = 0; i < nv; i++) similarity_table[i * nv + i] = 1;
  float diff;
  do {
    diff = 0;
    for (int i = 0; i < nv; i++) {
      if (i % 100 == 0) printf("i: %d max diff: %f\n", i, diff);
      for (int j = 0; j < nv; j++) {
        if (i == j) continue;
        float newsim = 0;
        for (int ia = offsets[i]; ia < offsets[i + 1]; ia++)
          for (int ja = offsets[j]; ja < offsets[j + 1]; ja++) {
            newsim += similarity_table[edges[ia] * nv + edges[ja]];
          }
        newsim *= c / (degree(i) * degree(j));
        diff = max(fabs(similarity_table[i * nv + j] - newsim), diff);
        similarity_table[i * nv + j] = newsim;
      }
    }
  } while (diff > 1e-4);
}

void deepwalksim(int walk_num, int walk_len, int window) {
  MKL_INT* walk = (MKL_INT*)malloc(sizeof(MKL_INT)*walk_len);
  for (int i = 0; i < nv; i++)
  {
	if(i%100 == 0)
	  printf("diff: %d \n", i);
	for (int j = 0; j < walk_num; j++) {
	  walk[0] = i;
	  for(int k=1;k<walk_len;k++)
		walk[k] = edges[FastRandrange(offsets[walk[k-1]], offsets[walk[k - 1] + 1])];
	  for(int k=0;k<walk_len;k++)
		for (int l = max(0, k - window); l < min(k + window, walk_len); l++)
		{
		  //if (k == l) continue; //TESTME
		  similarity_table[walk[k] * nv + walk[l]]++;
		  similarity_table[walk[l] * nv + walk[k]]++;
		}
	}
  }
  for (int i = 0; i < nv; i++)
  {
	float sum = cblas_sasum(nv, &similarity_table[i*nv], 1);
	cblas_sscal(nv, 1 / sum, &similarity_table[i*nv], 1);
  }
}

// DO NOT USE. Computes SimRank in a second dumbest possible way.
void simrank2() {
  for (int i = 0; i < nv; i++) similarity_table[i * nv + i] = 1;
  partials_table = (float*)mkl_malloc(nv * nv * sizeof(float), 64);
  do {
    memset(partials_table, 0, nv * nv * sizeof(float));
#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < nv; i++) {
	  for (int ia = offsets[i]; ia < offsets[i + 1]; ia++) {
		int j = edges[ia];
		for (int k = 0; k < nv; k++)
		  partials_table[i * nv + k] += similarity_table[j * nv + k];
	  }
	}
    sr_diff = 0;
#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < nv; i++) {
	  for (int j = 0; j < nv; j++) {
		if (i == j) continue;
		float newsim = 0;
		for (int ja = offsets[j]; ja < offsets[j + 1]; ja++) {
		  newsim += partials_table[i * nv + edges[ja]];
		}
		newsim *= sr_c / (degree(i) * degree(j));
		sr_diff = max(fabs(similarity_table[i * nv + j] - newsim), sr_diff);
		similarity_table[i * nv + j] = newsim;
	  }
	}
    printf("diff: %f \n", sr_diff);
  } while (sr_diff > 1e-6);
  mkl_free(partials_table);
}

// Helper function. Computes the degree of the node.
int degree(int node) { return offsets[node + 1] - offsets[node]; }

// Computes normalized Jaccard similarity from node i.
void jaccard(int i, float* sim) {
  sim[i] = 1;
  for (int j = offsets[i]; j < offsets[i + 1]; j++) {
    int res_and = 0;
    int res_or = 0;
    common_neighbors(i, edges[j], &res_or, &res_and);
    sim[edges[j]] = res_and / (float)res_or;
  }
  float sum = cblas_sasum(nv, sim, 1);
  cblas_sscal(nv, 1 / sum, sim, 1);
}

// Computes normalized cosine similarity from node i.
void cosine(int i, float* sim) {
  sim[i] = 1;
  for (int j = offsets[i]; j < offsets[i + 1]; j++) {
    int res_and = 0;
    int res_or = 0;
    common_neighbors(i, edges[j], &res_or, &res_and);
    sim[edges[j]] = res_and / sqrt(degree(i) * degree(edges[j]));
  }
  float sum = cblas_sasum(nv, sim, 1);
  cblas_sscal(nv, 1 / sum, sim, 1);
}

// Computes normalized neighbors node i.
void neigh(int i, float* sim) {
  sim[i] = degree(i);
  for (int j = offsets[i]; j < offsets[i + 1]; j++) {
    sim[edges[j]] = alpha / sim[i];
  }
  sim[i] = (1 - alpha) / sim[i];
}

// Computes normalized Adamic-Adar similarity from node i.
void adamicadar(int i, float* sim) {
  for (int j = offsets[i]; j < offsets[i + 1]; j++) {
    sim[i] += 1 / log(max(degree(edges[j]), 2));
    int i_s = offsets[i];
    int i_e = offsets[i + 1];
    int j_s = offsets[edges[j]];
    int j_e = offsets[edges[j] + 1];
    while (i_s < i_e || j_s < j_e) {
      int i_c = edges[i_s];
      int j_c = edges[j_s];
      if (i_c > j_c) {
        if (j_s < j_e)
          j_s++;
        else
          i_s++;
      } else if (i_c < j_c) {
        if (i_s < i_e)
          i_s++;
        else
          j_s++;
      } else {
        sim[edges[j]] += 1 / log(max(degree(i_c), 2));
        if (j_s < j_e) j_s++;
        if (i_s < i_e) i_s++;
      }
    }
  }
  float sum = cblas_sasum(nv, sim, 1);
  cblas_sscal(nv, 1 / sum, sim, 1);
}

// Helper function for cross-platform argument parsing.
int ArgPos(char* str, int argc, char** argv) {
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

int main(int argc, char** argv) {
  int a;
  int seed = 0;
  strcpy(optimizer_name, "adam");
  vmlSetMode(VML_EP | VML_FTZDAZ_ON);  // Enchanted performance
  srand(time(NULL));
  if (argc == 1) {
    printf("VERSE: VERtex Similarity Embeddings\n\n");
    printf("Program options:\n");
    printf("\t-input <file>\n");
    printf("\t\tUse network data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the embeddings\n");
    printf("\t-dim <int>\n");
    printf("\t\tSet dimension of vertex embeddings (default 128)\n");
    printf("\t-opt <string>\n");
    printf(
        "\t\tSet the optimizer to use from [sgd, momentum, nag, adam] (default "
        "adam)\n");
    printf("\t-sim <string>\n");
    printf(
        "\t\tSet the similarity measure to use from [ppr, neigh, adamic, "
        "cosine, jaccard] (default ppr)\n");
    printf("\t-lr <float>\n");
    printf("\t\tSet the initial learning rate (default 0.002)\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the alpha parameter for PPR (default 0.9)\n");
    printf("\t-sr_c <float>\n");
    printf("\t\tSet the SimRank c parameter (default 0.9)\n");
    printf("\t-precompute <0/1>\n");
    printf(
        "\t\tSet whether or not to precompute the similarity function (default "
        "0)\n");
    printf("\t-sampling <int>\n");
    printf(
        "\t\tSet the number of samples for similarity estimation. Set 0 to "
        "perform exact computation (default 3000)\n");
    printf("\t-seed <int>\n");
    printf(
        "\t\tSet the random seed. Set 0 for a random initialization (default "
        "0)\n");
    printf("\t-threads <int>\n");
    printf("\t\tSet the number of threads (default 1)\n");
    printf("\t-batch <int>\n");
    printf("\t\tSet the batch size (default 128)\n");
    printf("\nExample:\n");
    printf("./verse -input net.bcsr -output vec.bin -size 128 -lr 0.003\n\n");
    return 0;
  }
  if ((a = ArgPos((char*)"-input", argc, argv)) > 0)
    strcpy(network_file, argv[a + 1]);
  else {
    printf("Input file not given! Aborting now..");
    return 0;
  }
  if ((a = ArgPos((char*)"-output", argc, argv)) > 0)
    strcpy(embedding_file, argv[a + 1]);
  else {
    printf("Output file not given! Aborting now..");
    return 0;
  }
  if ((a = ArgPos((char*)"-opt", argc, argv)) > 0)
    strcpy(optimizer_name, argv[a + 1]);
  if ((a = ArgPos((char*)"-sim", argc, argv)) > 0)
    strcpy(similarity_name, argv[a + 1]);
  if ((a = ArgPos((char*)"-dim", argc, argv)) > 0) n_hidden = atoi(argv[a + 1]);
  if ((a = ArgPos((char*)"-verbose", argc, argv)) > 0) verbose = atoi(argv[a + 1]);
  if ((a = ArgPos((char*)"-epochs", argc, argv)) > 0) n_epochs = atoi(argv[a + 1]);
  if ((a = ArgPos((char*)"-lr", argc, argv)) > 0) global_lr = atof(argv[a + 1]);
  if ((a = ArgPos((char*)"-alpha", argc, argv)) > 0) alpha = atof(argv[a + 1]);
  if ((a = ArgPos((char*)"-sr_c", argc, argv)) > 0) sr_c = atof(argv[a + 1]);
  if ((a = ArgPos((char*)"-precompute", argc, argv)) > 0)
    precompute_ppr = atoi(argv[a + 1]);
  if ((a = ArgPos((char*)"-sampling", argc, argv)) > 0) {
    n_samples = atoi(argv[a + 1]);
    use_sampling = n_samples > 0;
  }
  if ((a = ArgPos((char*)"-seed", argc, argv)) > 0) seed = atoi(argv[a + 1]);
  if ((a = ArgPos((char*)"-threads", argc, argv)) > 0)
    num_threads = atoi(argv[a + 1]);
  if ((a = ArgPos((char*)"-batch", argc, argv)) > 0)
    n_batch = atoi(argv[a + 1]);

  printf("Using input file: %s, writing to %s. lr=%f\n", network_file,
         embedding_file, global_lr);
  printf("\n");
  if (seed == 0) seed = rand();
  if (strcmp(similarity_name, "adamic") == 0) {
    similarity = &adamicadar;
  } else if (strcmp(similarity_name, "cosine") == 0) {
    similarity = &cosine;
  } else if (strcmp(similarity_name, "jaccard") == 0) {
    similarity = &jaccard;
  } else if (strcmp(similarity_name, "neigh") == 0) {
    similarity = &cosine;
  } else {
    if (use_sampling)
      similarity = &estimate_ppr_rw;
    else
      similarity = &estimate_ppr_poweriter_csr;
  }
  vslNewStream(&rng_stream, VSL_BRNG_SFMT19937, seed);
  InitRandomTable(&rng_stream);
  FILE* fptr = fopen(network_file, "rb");
  char header[] = "----";
  fread(header, sizeof(char), 4, fptr);
  if (strcmp(header, "XGFS") != 0) {
    printf("Invalid header! %s\n", header);
    getchar();
    return 1;
  }
  fread(&nv, sizeof(MKL_INT64), 1, fptr);
  fread(&ne, sizeof(MKL_INT64), 1, fptr);
  r_offsets = (int*)mkl_malloc((nv + 1) * sizeof(int), 64);
  r_edges = (int*)mkl_malloc(ne * sizeof(int), 64);
  offsets = (MKL_INT*)mkl_malloc((nv + 1) * sizeof(MKL_INT), 64);
  edges = (MKL_INT*)mkl_malloc(ne * sizeof(MKL_INT), 64);
  printf("nv=%ld, ne=%ld\n", nv, ne);
  fread(r_offsets, sizeof(int), (size_t)nv, fptr);
  r_offsets[nv] = (int)ne;
  fread(r_edges, sizeof(int), (size_t)ne, fptr);
  for (int i = 0; i < nv + 1; i++) offsets[i] = r_offsets[i];
  for (int i = 0; i < ne; i++) edges[i] = r_edges[i];
  fclose(fptr);
  if (!use_sampling) {
    elems = (float*)mkl_malloc(ne * sizeof(float), 64);
    int edgeidx = 0;
    for (int i = 0; i < nv; i++)
      for (int j = offsets[i]; j < offsets[i + 1]; j++)
        elems[edgeidx++] = 1.f / (offsets[i + 1] - offsets[i]);
  }
#if USE_PR_WEIGHTS
  prs = (float*)mkl_malloc(nv * sizeof(float), 64);
  printf("Computing PageRank vertex improtances..\n");
  if (use_sampling) {
    estimate_pr_rw(prs);
  } else {
    estimate_pr_poweriter_csr(prs);
  }
  cblas_sscal(nv, (float)nv, prs, 1);
  for (int i = 0; i < nv; i++) prs[i] = min(prs[i], 1);
#endif
  w0 = (float*)mkl_malloc(nv * n_hidden * sizeof(float), 64);
  float init_width = 1.0 / sqrt(n_hidden);
  vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, rng_stream, nv * n_hidden,
                w0, 0, init_width);
  printf("Using %s optimizer\n", optimizer_name);  
  printf("Precompute set to %d, sampling set to %d.\n", precompute_ppr,
         use_sampling);
  std::chrono::steady_clock::time_point begin;
  std::chrono::steady_clock::time_point end;
  if (precompute_ppr) {
    similarity_table = (float*)mkl_malloc(nv * nv * sizeof(float), 64);
	 begin = std::chrono::steady_clock::now();
#pragma omp parallel num_threads(num_threads)
	{
	  int tid = omp_get_thread_num();
#pragma omp for
	  for (int i = 0; i < nv; i++) {
		ppr_step++;
		if(tid==0)
		  printf("\rPrecomputation.. %3.2f%%", ppr_step * 100. / nv);
		(*similarity)(i, &similarity_table[i * nv]);
	  }
	}
    printf("\n");
	end = std::chrono::steady_clock::now();
    printf("\nPrecomputation took %.2lf s to run.\n", std::chrono::duration_cast<std::chrono::duration<float>>(end - begin).count());
  }
  total_steps = nv * n_epochs / n_batch;
  begin = std::chrono::steady_clock::now();
  Train();
  end = std::chrono::steady_clock::now();
  printf("\n");
  printf("\n\nCalculations took %.2lf s to run.\n", std::chrono::duration_cast<std::chrono::duration<float>>(end - begin).count());
  if (w0[0] != w0[0]) return 1;
  fptr = fopen(embedding_file, "wb");
  fwrite(w0, sizeof(float), (size_t)nv * n_hidden, fptr);
  fclose(fptr);
}

void Shuffle(int* a, int n) {
  int i, j, temp;
  for (i = n - 1; i >= 0; i--) {
    j = FastRandrange(0, i + 1);
    temp = a[j];
    a[j] = a[i];
    a[i] = temp;
  }
}

void Train() {
#pragma omp parallel num_threads(num_threads)
  {
	int tid = omp_get_thread_num();
	float one = 1;
	float* dw0 = (float*)mkl_malloc(nv * n_hidden * sizeof(float), 64);
	void* dw0_p;
	if (strcmp(optimizer_name, "adam") ==
	  0) {  // This is how overloading looks like
	  optimizer = &Adam;
	  dw0_p = malloc(sizeof(AdamParams));
	  ((AdamParams*)dw0_p)->m =
		(float*)mkl_malloc(nv * n_hidden * sizeof(float), 64);
	  ((AdamParams*)dw0_p)->v =
		(float*)mkl_malloc(nv * n_hidden * sizeof(float), 64);
	}
	else if (strcmp(optimizer_name, "momentum") == 0) {
	  optimizer = &Momentum;
	  dw0_p = malloc(sizeof(MomentumParams));
	  ((MomentumParams*)dw0_p)->v =
		(float*)mkl_malloc(nv * n_hidden * sizeof(float), 64);
	}
	else if (strcmp(optimizer_name, "nesterov") == 0) {
	  optimizer = &Momentum;
	  dw0_p = malloc(sizeof(NAGParams));
	  ((NAGParams*)dw0_p)->v =
		(float*)mkl_malloc(nv * n_hidden * sizeof(float), 64);
	}
	else {
	  optimizer = &SGD;
	}
	float* dhidden = (float*)mkl_malloc(n_batch * n_hidden * sizeof(float), 64);
	float* hidden = (float*)mkl_malloc(n_batch * n_hidden * sizeof(float), 64);
	float* scores = (float*)mkl_malloc(n_batch * nv * sizeof(float), 64);
	float* scores_exp = (float*)mkl_malloc(n_batch * nv * sizeof(float), 64);
	float* output_exp = (float*)mkl_malloc(n_batch * sizeof(float), 64);

	float* pprs;
	if (!precompute_ppr) pprs = (float*)mkl_malloc(nv * sizeof(float), 64);

#if !USE_EMBEDDING_LOOKUP
	float* x = (float*)mkl_malloc(nv * n_batch * sizeof(float), 64);
#endif
	int* xid = (int*)mkl_malloc(n_batch * sizeof(int), 64);
	float* y = (float*)mkl_malloc(nv * n_batch * sizeof(float), 64);

	int* order = (int*)mkl_malloc(nv * sizeof(int), 64);
	for (int i = 0; i < nv; i++) order[i] = i;
	Shuffle(order, nv);
	do {
	  int local_step = step;  // multitreading stuff
	  float local_lr = global_lr;
#if USE_EMBEDDING_LOOKUP
	  memset(dw0, 0, nv * n_hidden * sizeof(float));
#else
	  memset(x, 0, nv * n_batch * sizeof(float));
#endif
	  for (int i = 0; i < n_batch; i++) {
		xid[i] = order[(local_step * n_batch + i) % nv];
#if !USE_EMBEDDING_LOOKUP
		x[i * nv + xid[i]] = 1;
#endif
		if (precompute_ppr) {
		  cblas_scopy(nv, &similarity_table[xid[i] * nv], 1, &y[i * nv], 1);
		}
		else {
		  memset(pprs, 0, nv * sizeof(float));
		  if (!use_sampling) {
			(*similarity)(xid[i], pprs);
		  }
		  else {
			(*similarity)(xid[i], pprs);
		  }
		  cblas_scopy(nv, pprs, 1, &y[i * nv], 1);
		}
	  }
#if !USE_EMBEDDING_LOOKUP
	  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_batch, n_hidden,
		nv, 1, x, nv, w0, n_hidden, 0, hidden, n_hidden);
#else
	  for (int i = 0; i < n_batch; i++)
		cblas_scopy(n_hidden, &w0[xid[i] * n_hidden], 1, &hidden[i * n_hidden],
		  1);
#endif
	  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_batch, nv, n_hidden,
		1, hidden, n_hidden, w0, n_hidden, 0, scores, nv);
	  memset(output_exp, 0, n_batch * sizeof(float));
#if USE_TANH_NONLINEARITY
	  vsTanh(n_batch * nv, scores, scores);
#endif
	  vsExp(n_batch * nv, scores, scores_exp);
#if USE_TANH_NONLINEARITY
	  vsSqr(n_batch * nv, scores, scores);
	  cblas_saxpby(n_batch * nv, 1, &one, 0, -1, scores, 1);
#endif
	  for (int i = 0; i < n_batch; i++)
		output_exp[i] = cblas_sasum(nv, &scores_exp[i * nv], 1);
	  for (int i = 0; i < n_batch; i++)
		cblas_sscal(nv, 1 / output_exp[i], &scores_exp[i * nv], 1);
	  float loss = 0;

	  for (int i = 0; i < n_batch * nv; i++)
		loss -= log(scores_exp[i] + 1e-10) * y[i];
	  loss /= n_batch;

	  printf("\rProgress %3.2f%%", 100.0 * local_step / total_steps);
	  if (loss != loss) {
		printf("\nNaN loss! Aborting..");
		break;
	  }
	  for (int i = 0; i < n_batch; i++)
		vsSub(nv, &scores_exp[i * nv], &y[i * nv], &scores_exp[i * nv]);
	  cblas_sscal(n_batch * nv, 1. / n_batch, scores_exp, 1);
#if USE_PR_WEIGHTS
	  for (int i = 0; i < n_batch; i++)
		cblas_sscal(nv, prs[xid[i]], &scores_exp[i * nv], 1);
#endif
#if USE_TANH_NONLINEARITY
	  vsMul(n_batch * nv, scores, scores_exp, scores);
#else
	  cblas_scopy(n_batch * nv, scores_exp, 1, scores, 1);
#endif
	  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_batch, n_hidden,
		nv, 1, scores, nv, w0, n_hidden, 0, dhidden, n_hidden);
#if !USE_EMBEDDING_LOOKUP
	  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nv, n_hidden, n_batch,
		1, x, nv, dhidden, n_hidden, 0, dw0, n_hidden);
#else
	  for (int i = 0; i < n_batch; i++)
		cblas_scopy(n_hidden, &dhidden[i * n_hidden], 1, &dw0[xid[i] * n_hidden],
		  1);
#endif
	  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nv, n_hidden, n_batch,
		1, scores, nv, hidden, n_hidden, 1, dw0, n_hidden);
	  (*optimizer)(nv * n_hidden, local_lr, w0, dw0, dw0_p);
#pragma omp atomic
	  step++;
	} while (step < total_steps);
	mkl_free(dw0);
	mkl_free(dhidden);
	mkl_free(hidden);
	mkl_free(scores);
	mkl_free(output_exp);
	if (strcmp(optimizer_name, "adam") == 0) {
	  mkl_free(((AdamParams*)dw0_p)->m);
	  mkl_free(((AdamParams*)dw0_p)->v);
	  free(dw0_p);
	}
	else if (strcmp(optimizer_name, "momentum") == 0) {
	  mkl_free(((MomentumParams*)dw0_p)->v);
	  free(dw0_p);
	}
	else if (strcmp(optimizer_name, "nesterov") == 0) {
	  mkl_free(((NAGParams*)dw0_p)->v);
	  free(dw0_p);
	}
	if (!precompute_ppr) mkl_free(pprs);
  }
}

void SGD(int sz, float lr, float* w, float* dw, void* params) {
  cblas_sscal(sz, lr, dw, 1);
  // w <- w - lr * dw
  vsSub(sz, w, dw, w);
}

void Momentum(int sz, float lr, float* w, float* dw, void* params) {
  float* v = ((MomentumParams*)params)->v;
  cblas_sscal(sz, momentum_1, v, 1);
  cblas_sscal(sz, lr, dw, 1);
  // w <- w + m * v - lr * dw
  vsSub(sz, v, dw, v);
  vsAdd(sz, w, dw, w);
}

void NAG(int sz, float lr, float* w, float* dw, void* params) {
  float* v = ((NAGParams*)params)->v;
  cblas_sscal(sz, momentum_1, v, 1);
  cblas_sscal(sz, lr, dw, 1);
  vsSub(sz, v, dw, v);
  // w <- w - m * m * v - (1 + m) * lr * dw
  cblas_saxpby(sz, -momentum_1 * momentum_1, v, 1, 1 + momentum_1, dw, 1);
  vsSub(sz, w, dw, w);
}

float Adam_lr(float lr) {
  float fix1 = 1 - pow(momentum_1, step + 1);
  float fix2 = 1 - pow(momentum_2, step + 1);
  return lr * sqrt(fix2) / fix1;
}

void Adam(int sz, float lr, float* w, float* dw, void* params) {
  float* v = ((AdamParams*)params)->v;
  float* m = ((AdamParams*)params)->m;
  // m_t <- b1 * m_{t-1} + (1 - b1) * g
  cblas_sscal(sz, momentum_1, m, 1);
  cblas_sscal(sz, 1 - momentum_1, dw, 1);
  vsAdd(sz, m, dw, m);
  cblas_sscal(sz, 1 / (1 - momentum_1), dw, 1);
  // v_t < - b2 * v_{t - 1} + (1 - b2) * g * g
  vsSqr(sz, dw, dw);
  cblas_sscal(sz, 1 - momentum_2, dw, 1);
  vsAdd(sz, v, dw, v);
  // w <- w - lr_t * m_t / (sqrt(v_t) + e)
  float local_lr = Adam_lr(lr);
  vsSqrt(sz, v, dw);
  cblas_saxpy(sz, 1, &epsilon, 0, dw, 1);
  vsDiv(sz, m, dw, dw);
  cblas_sscal(sz, local_lr, dw, 1);

  vsSub(sz, w, dw, w);
}