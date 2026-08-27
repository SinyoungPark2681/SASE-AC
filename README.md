# Squared Adjacency Spectral Embedding for Covariate-Assisted Community Detection

## Abstract
Community detection has been studied broadly since spectral clustering was introduced, and various spectral methods and their extensions have been developed. In addition to methods that identify community structures using only network information, clustering methods that incorporate node covariates have also been introduced. More recently, network generation models have been developed to incorporate covariates directly into the network generation. However, most of these methods have mainly focused on assortative and non-assortative networks. However, another important network structure is core-periphery. To provide a more robust approach for community detection, we propose a new method that incorporates the idea of the squared adjacency matrix into the additive-covariate stochastic block model. In this framework, covariates are used to generate networks, while the squared adjacency structure is used to identify the underlying communities. The proposed approach is designed to adapt to both non-assortative and core-periphery networks. We demonstrate its stable clustering performance across varying covariate effects through simulation studies. We also demonstrate the clustering performance of the proposed method on real-world datasets.

## Authors
[Sinyoung Park](https://sinyoungpark2681.github.io/), [Matthew Nunes](https://people.bath.ac.uk/man54/homepage.html), [Sandipan Roy](https://sites.google.com/view/sandipanroy)

## Codes
The files, generation.py, models.py, utils.py in the ACSBM folder, are adapted from the code accompanying the paper, [Perfect Spectral Clustering with Discrete Covariates](https://www3.stat.sinica.edu.tw/sstest/j35n31/j35n3105/j35n3105.html). The original implementation is available in authors' [GitHub repository](https://github.com/jonhehir/acsbm).
The file, estimation.py in ACSBM folder, is based on the original implementation and has been modified to incorporate squared adjacency spectral embedding.
