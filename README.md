# VERSE

VERtex Similarity Embedding

## Requirements

Intel C++ compiler for VERSE, any C++ compiler with OpenMP for sVERSE.

## Usage

We provide binaries in the `bin` folder for AVX instruction set.

### Full VERSE

`verse -input data/vk.bcsr -output embs.bin -threads 4 -sim ppr -precompute 0 -sampling 0`

Available similarities:

* Adamic-Adar
* Cosine similarity
* Adjacency similarity
* Jaccard index
* DeepWalkSim (through code edit)
* SimRank (through code edit)

### sVERSE

`sverse-ppr -input data/blogcatalog.bcsr -output embs.bin -threads 4 -dim 128`