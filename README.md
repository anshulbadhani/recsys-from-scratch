# recsys-from-scratch

A high-performance Two-Tower Recommendation Engine built entirely from scratch in C++ – no external libraries, no frameworks, no shortcuts.

**354× faster** than the unoptimized baseline. **0.33ms per user**. **81.18% catalog coverage**.

## What it does

Given a user's interaction history, the system retrieves and ranks a list of items they haven't seen, ordered by predicted relevance. Evaluated on 146,980 users from the Amazon Software dataset under strict leave-one-out protocol.

The interesting part isn't the recommendation quality. It's that production-grade inference latency is achievable from first principles, without calling `faiss.index.search()` or any other library that hides what's actually happening.


## Results

| Configuration | Latency | Speedup | Coverage |
|---|---|---|---|
| Baseline (no opt.) | 116.8ms/user | 1× | — |
| + Compiler (-O3) | 35.0ms/user | 3.34× | — |
| + Candidate reduction | 31.7ms/user | 3.68× | — |
| + AVX2 SIMD | 15.45ms/user | 7.56× | — |
| + OpenMP (14 cores) | 4.36ms/user | 26.8× | — |
| **+ PCA (384D → 64D)** | **0.33ms/user** | **354×** | — |

| Method | Recall@10 | NDCG@10 | Coverage |
|---|---|---|---|
| Popularity Baseline | 0.0650 | 0.0315 | 0.01% |
| Cosine Sort (384D) | 0.0168 | 0.0088 | 19.46% |
| Adaptive MMR (384D) | 0.0118 | 0.0072 | 37.28% |
| Cosine Sort (64D, signed) | 0.0124 | 0.0063 | 70.42% |
| **Adaptive MMR (64D, signed)** | **0.0067** | **0.0043** | **81.18%** |

The Popularity Baseline achieves the highest Recall by recommending the same 9 items to all 146,980 users. That's not a recommendation system – that's dataset memorization.


## Architecture
<img width="558" height="433" alt="inference_pipeline" src="https://github.com/user-attachments/assets/f7c71418-f7ac-4e11-9d04-a246c9aefa45" />
<br><br>

**Offline (Python):** MiniLM embeddings → PCA reduction (384D → 64D) → artifacts for C++ pipeline

**Online (C++, zero dependencies):**
- **User Embedding:** bipolar rating weights mapping 1★→−1, 3★→0, 5★→+1, projected onto unit hypersphere
- **KD-Tree:** custom ANN retrieval, O(log N) with PCA preprocessing, built with `std::nth_element` (QuickSelect)
- **Bloom Filter:** per-user seen-item filter, k=7 hash functions via double hashing, p=0.01 false positive rate
- **Adaptive MMR:** novel extension of Maximal Marginal Relevance with position-dependent λ decay, unifying cosine sort and standard MMR as special cases



## Novel Contributions

**Adaptive MMR:** standard MMR applies the same λ at every position. This ignores that early recommendations should exploit relevance while later ones should explore diversity. We introduce position-dependent exponential decay:

```
λ_pos = λ_max × e^(−δ × pos)
```

Cosine sort (λ=1, δ=0) and standard MMR (λ=0.5, δ=0) are both special cases.

**Bipolar signed weights:** mapping ratings to [−1, 1] instead of (0, 1] lets low ratings repel the user vector, dispersing it across the embedding space. Result: catalog coverage jumps from 17% to 70% for cosine sort, 26% to 81% for Adaptive MMR. One formula, 4× coverage improvement.



## The optimization that mattered most

PCA from 384D to 64D delivered a **13.2× single-step speedup**, the largest in the pipeline.

At 384 dimensions, the KD-tree degrades to O(N) because all points become nearly equidistant (Beyer et al., 1999). The pruning condition almost never fires. The embedding matrix occupies 137MB, far exceeding L3 cache. Every node access is a main memory fetch at 70–100 cycle latency.

At 64 dimensions, the matrix shrinks to ~23MB, the upper tree levels become L3-resident after warmup, and spatial pruning fires again. The quality cost: Recall@10 drops from 0.0168 to 0.0146. The speed gain: 13.2×.


## Running

**Requirements:** Docker, Docker Compose, AVX2-capable CPU (Intel Haswell or newer)

```bash
# Build and start container
docker-compose build
docker-compose up -d
docker-compose exec recsys bash
```

```bash
# Inside container: run offline preprocessing (Python)
cd scripts
bash run_scripts.bash

# -- OR --
uv sync
uv run <script_name.py>

# Build and run C++ inference engine
cd ..
mkdir -p build && cd build && cmake .. && make && cd ..
./build/bin/main
```

Data is downloaded automatically by `scripts/01_download_data.py` from the Amazon Reviews 2023 dataset (Software category).


## Project structure

```bash
recsys-from-scratch/
├── CMakeLists.txt
├── Dockerfile
├── data                                # data
│   ├── embeddings
│   │   ├── item_embedding_index.csv
│   │   ├── item_embeddings copy.npy
│   │   ├── item_embeddings.csv
│   │   └── item_embeddings.npy
│   ├── meta_Software.jsonl
│   ├── metadata_Software.jsonl.gz
│   ├── models
│   ├── reviews_Software.jsonl
│   ├── reviews_Software.jsonl.gz
├── docker-compose.yml
├── scripts
│   ├── 00_category_selection.py        # To study different data categories
│   ├── 01_download_data.py
│   ├── 02_check_data.py
│   ├── 03_filter_data.py
│   ├── 04_item_embedding.py            # Generates embeddings using miniLM
│   ├── 04b_apply_pca.py                # 384D -> 64D
│   ├── 05_evaluate.py                  # To calculate performance metrics from generated result files
│   └── config.py                       # all the settings for python scripts
├── include
│   ├── bloom_filter.h
│   ├── config.h                        # settings for C++ files
│   ├── data_loader.h                   # csv parser
│   ├── kdtree.h
│   ├── pipeline.h                      # to glue all components together
│   ├── ranker.h
│   └── user_embedding.h                # generates user embeddings
└── src                                 # C++ source files
    ├── embeddings
    │   └── user_embedding.cpp
    ├── main.cpp                        # program entry point
    ├── pipeline.cpp
    ├── ranking
    │   └── ranker.cpp
    ├── retrieva
    │   ├── bloom_filter.cpp
    │   └── kdtree.cpp
    └── utils
        └── data_loader.cpp
```


## Future work

- **HNSW:** graph-based ANN that achieves O(log N) regardless of dimensionality, eliminating the need for PCA and recovering the Recall@10 lost to dimensionality reduction
- **LinUCB:** online feedback loop with provably sublinear regret O(d√T log T), the system currently has no way to adapt to evolving preferences
- **SASRec:** self-attention over interaction sequences, capturing order signals the current bag-of-items embedding explicitly discards
