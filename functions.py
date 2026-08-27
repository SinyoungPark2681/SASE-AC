from collections import defaultdict
from scipy.stats import bernoulli

from acsbm import generation, estimation, utils
from acsbm import models
from acsbm.examples import run_simulation

import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import time
from tqdm.notebook import tqdm
import scipy.linalg as sp
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
import math
from scipy import sparse

def diag(A):
    
    D = np.zeros((len(A), len(A)))
    for i in range(len(A)):
        D[i,i] = sum(A[i,:])
        
    return D


def make_labels(n, cluster_sizes):
    """
    Generate cluster labels for a network with k clusters.

    Returns labels in 0-based indexing: 0,1,2,...
    """
    
    if sum(cluster_sizes) != n:
        raise ValueError("Sum of cluster sizes must equal the total number of nodes")

    total_nodes = list(range(n))
    clusters = []
    labels = []

    start_index = 0
    for cluster_id, size in enumerate(cluster_sizes):   # <-- start at 0
        end_index = start_index + size
        cluster_nodes = total_nodes[start_index:end_index]
        clusters.append(cluster_nodes)
        labels.extend([cluster_id] * size)              # <-- now 0,1,2,...
        start_index = end_index

    return clusters, labels


def generate_X_with_sizes(sizes, M, seed=None):
    import numpy as np

    # random generator
    rng = np.random.default_rng(seed)

    K = len(sizes)
    n = sum(sizes)

    # group labels
    z = np.repeat(np.arange(K), sizes)
    rng.shuffle(z)

    if len(sizes) == 2:
        m1, m2 = M[0], M[1]

        # probability matrix
        M = np.array([
            [m1, 1-m2],
            [1-m1, m2]
        ])
    
        # generate X
        X = np.zeros((n, 2), dtype=int)
    
        for i in range(n):
            for r in range(2):
                X[i, r] = rng.binomial(1, M[z[i], r])

    if len(sizes) == 3:
        m1, m2, m3 = M[0], M[1], M[2]

        # probability matrix
        M = np.array([
            [m1, 1-m2, 1-m3],
            [1-m1, m2, 1-m3],
            [1-m1, 1-m2, m3]
        ])
    
        # generate X
        X = np.zeros((n, 3), dtype=int)
    
        for i in range(n):
            for r in range(3):
                X[i, r] = rng.binomial(1, M[z[i], r])

    return X


def generate_X_with_sizes2(sizes, M, seed=None):
    import numpy as np

    rng = np.random.default_rng(seed)

    K = len(sizes)
    n = sum(sizes)

    z = np.repeat(np.arange(K), sizes)
    rng.shuffle(z)

    m1, m2 = M[0], M[1]

    M_prob = np.array([
        [m1, 1-m2],
        [1-m1, m2],
        [1-m1, 1-m2]
    ])

    X = np.zeros((n, 2), dtype=int)

    for i in range(n):
        for r in range(2):
            X[i, r] = rng.binomial(1, M_prob[z[i], r])

    return X


def cov_function(X, beta):
    n = len(X)
    cov = np.zeros((n,n))
    m_len = len(beta)
    
    for m in range(m_len):
        cov1 = np.subtract.outer(X[:,m], X[:,m])
        cov1 = (cov1 == 0)

        cov1 = cov1 * beta[m]
        cov += cov1
        
    return cov


def Sim_ACSBM2(n, beta, X, sparsity, relsize, relb, relw, k, seed=None, direction=False):
    import numpy as np
    import math

    if seed is not None:
        np.random.seed(seed)

    # community sizes
    sizes = [math.ceil(rels * n) for rels in relsize]
    sizes[-1] = n - sum(sizes[:-1])

    # block labels, assuming nodes are ordered by community
    labels = np.repeat(np.arange(k), sizes)

    # k x k log-scale relative block matrix W
    W = np.zeros((k, k))

    for a in range(k):
        W[a, a] = relw[a]

    for a in range(k):
        for b in range(a + 1, k):
            W[a, b] = relb[a][b]
            W[b, a] = relb[a][b]

    # node-level log-scale block effect matrix
    B_weight = W[labels[:, None], labels[None, :]]

    # covariate effect matrix:
    # cov_matrix[i,j] = sum_r beta_r I(X_ir = X_jr)
    cov_matrix = cov_function(X, beta)

    # remove diagonal effects
    np.fill_diagonal(B_weight, 0)
    np.fill_diagonal(cov_matrix, 0)

    # unscaled probability component under log link
    R = np.exp(B_weight + cov_matrix)
    np.fill_diagonal(R, 0)

    # scaling factor to match target expected density
    if direction:
        # directed graph: use all ordered pairs i != j
        idx = np.where(~np.eye(n, dtype=bool))
        T = n * (n - 1)
    else:
        # undirected graph: use unordered pairs i < j
        idx = np.triu_indices(n, 1)
        T = math.comb(n, 2)

    f = np.sum(R[idx])
    c = sparsity * T / f

    # final probability matrix
    P = c * R
    np.fill_diagonal(P, 0)

    if np.max(P) > 1:
        raise ValueError(
            f"Some probabilities exceed 1. max(P)={np.max(P):.4f}. "
            "Try smaller sparsity, smaller beta, or smaller block effects."
        )

    # log-scale block matrix for ACSBM model
    B_log = W + np.log(c)
    B_prob = np.exp(B_log)

    return P, B_prob, B_log, labels


def make_Z_tilde(X):
    _, Z_tilde = np.unique(X, axis=0, return_inverse=True)
    return Z_tilde


def process_node_iterations_ACSBM2_AA(node_list, iteration, relsize, beta, k, B0, M, link='log', direction=False, d=None, method='K-Means'):
    methods = ["acsbm1", "acsbm3", "acsbm1_aa", "acsbm3_aa"]

    dense_results = []
    NMI  = {m: [] for m in methods}
    STD  = {m: [] for m in methods}
    TIME = {m: [] for m in methods}

    progress_bar = tqdm(total=len(node_list) * iteration, desc="Processing")

    for n in node_list:

        n = int(n)
        cluster_sizes = [int(n*r) for r in relsize]
        _, labels = make_labels(n, cluster_sizes)   # zero-based labels
        
        density_list = []

        score = {m: [] for m in methods}
        tlist = {m: [] for m in methods}

        for it in range(iteration):

            # 1. Generate adjacency via block + covariates
            # B = np.exp(B) * (n**(-0.8))


            if k==2:
                B_scaled = B0 + np.log(n**(-0.8))
                model = models.MultiCovariateModel(
                    B = B_scaled,
                    covariates=[
                        models.Covariate.simple(beta[m], k) 
                        for m in range(M)
                    ],
                    link=models.LinkFunction.log()
                )
            elif k==3:
                B_scaled = B0 + np.log(n**(-0.5))
                model = models.MultiCovariateModel(
                    B = B_scaled,
                    covariates=[
                        models.Covariate.simple(beta[m], k) 
                        for m in range(M)
                    ],
                    link=models.LinkFunction.log()
                )

            ndd = models.NodeDataDistribution.uniform_for_model(model)
            net = generation.generate_network_directed(model, ndd, n, directed=direction, random_state=it)
            Adj = net.A.toarray()
            X = net.Z

            G = nx.from_numpy_matrix(Adj)
            density_list.append(nx.density(G))
            
            # ACSBM step1/step3
            t0 = time.time()
            ic = estimation.initial_cluster(net, k=model.n_communities, d=len(net.block_sizes), random_state=it)
            tlist["acsbm1"].append(time.time()-t0)
            score["acsbm1"].append(normalized_mutual_info_score(labels, ic.theta_tilde))

            t0 = time.time()
            rc = estimation.reconcile_clusters(net, ic)
            tlist["acsbm3"].append(time.time()-t0)
            score["acsbm3"].append(normalized_mutual_info_score(labels, rc.theta))

            # ACSBM_AA step1/step3
            net_aa = generation.GeneratedNetwork(
                A=Adj,
                theta=net.theta,
                Z=net.Z,
                Z_tilde=net.Z_tilde,
                block_sizes=net.block_sizes,
                model=net.model
            )

            # Step1 on AA
            t0 = time.time()
            ic_aa = estimation.initial_cluster_DA(net_aa, k=model.n_communities, d=len(net_aa.block_sizes), random_state=it)
            tlist["acsbm1_aa"].append(time.time() - t0)
            score["acsbm1_aa"].append(normalized_mutual_info_score(labels, ic_aa.theta_tilde))

            # Step3 on AA
            t0 = time.time()
            rc_aa = estimation.reconcile_clusters(net_aa, ic_aa)
            tlist["acsbm3_aa"].append(time.time() - t0)
            score["acsbm3_aa"].append(normalized_mutual_info_score(labels, rc_aa.theta))

            progress_bar.update(1)

        # ---- aggcore_mate per sparsity ----
        dense_results.append(np.mean(density_list))
        
        # ---- aggcore_mate per n ----
        for m in methods:
            NMI[m].append(np.mean(score[m]))
            STD[m].append(np.std(score[m]))
            TIME[m].append(np.mean(tlist[m]))

    progress_bar.close()
    print("Processing completed.")

    # ---- keep your original return order ----
    return (
        dense_results,
        NMI["acsbm1"], NMI["acsbm3"], NMI["acsbm1_aa"], NMI["acsbm3_aa"],
        STD["acsbm1"], STD["acsbm3"], STD["acsbm1_aa"], STD["acsbm3_aa"],
        TIME["acsbm1"], TIME["acsbm3"], TIME["acsbm1_aa"], TIME["acsbm3_aa"]
    )


def process_sparsity_iterations_ACSBM_AA(n, X, labels, sparsity_list, iteration, relsize, relw12, relb, beta, k, link='log', direction=False, method='K-Means'):
    
    methods = ["acsbm1", "acsbm3", "acsbm1_aa", "acsbm3_aa"]

    dense_results = []
    NMI  = {m: [] for m in methods}
    STD  = {m: [] for m in methods}
    TIME = {m: [] for m in methods}

    progress_bar = tqdm(total=len(sparsity_list) * iteration, desc="Processing")

    for sparsity in sparsity_list:
        density_list = []

        score = {m: [] for m in methods}
        tlist = {m: [] for m in methods}

        for j in range(iteration):
            # ---- generate network ----
            P, B_prob, B_log, labels_ordered = Sim_ACSBM2(
                n=n, beta=beta, X=X, sparsity=sparsity, relsize=relsize, relb=relb, relw=relw12, 
                k=k, seed=j, direction=direction
            )
            if k==2:
                model = models.MultiCovariateModel(
                    B=B_log,
                    covariates=[
                        models.Covariate.simple(beta[m], len(np.unique(X[:, m])))
                        for m in range(X.shape[1])],
                    link=models.LinkFunction.log())
            elif k==3:
                model = models.MultiCovariateModel(
                    B=B_log,
                    covariates=[
                        models.Covariate.simple(beta[m], len(np.unique(X[:, m])))
                        for m in range(X.shape[1])],
                    link=models.LinkFunction.log())

            triu = np.triu_indices(n, 1)
            Adj = np.zeros((n, n), dtype=int)
            Adj[triu] = np.random.binomial(1, P[triu])
            if direction==False:
                Adj = Adj + Adj.T

            G = nx.from_numpy_matrix(Adj)
            density_list.append(nx.density(G))

            Z_tilde = make_Z_tilde(X)

            # subcommunity labels: (theta, Z_tilde)
            n_z = len(np.unique(Z_tilde))
            sub_labels = labels_ordered * n_z + Z_tilde
            block_sizes = np.bincount(sub_labels, minlength=k * n_z)
            
            net = generation.GeneratedNetwork(
                A=sparse.csr_matrix(Adj.astype(float)),
                theta=labels_ordered,
                Z=X,
                Z_tilde=Z_tilde,
                block_sizes=block_sizes,
                model=model
            )

            # ACSBM step1/step3
            t0 = time.time()
            ic = estimation.initial_cluster(net, k=model.n_communities, d=len(net.block_sizes))
            tlist["acsbm1"].append(time.time()-t0)
            score["acsbm1"].append(normalized_mutual_info_score(labels, ic.theta_tilde))

            t0 = time.time()
            rc = estimation.reconcile_clusters(net, ic)
            tlist["acsbm3"].append(time.time()-t0)
            score["acsbm3"].append(normalized_mutual_info_score(labels, rc.theta))

            # ACSBM_AA step1/step3
            net_aa = generation.GeneratedNetwork(
                A=sparse.csr_matrix(Adj.astype(float)),
                theta=net.theta,
                Z=net.Z,
                Z_tilde=net.Z_tilde,
                block_sizes=net.block_sizes,
                model=net.model
            )

            # Step1 on AA
            t0 = time.time()
            ic_aa = estimation.initial_cluster_DA(net_aa, k=model.n_communities, d=len(net.block_sizes))
            tlist["acsbm1_aa"].append(time.time() - t0)
            score["acsbm1_aa"].append(normalized_mutual_info_score(labels, ic_aa.theta_tilde))

            # Step3 on AA
            t0 = time.time()
            rc_aa = estimation.reconcile_clusters(net_aa, ic_aa)
            tlist["acsbm3_aa"].append(time.time() - t0)
            score["acsbm3_aa"].append(normalized_mutual_info_score(labels, rc_aa.theta))

            progress_bar.update(1)

        # ---- aggcore_mate per sparsity ----
        dense_results.append(np.mean(density_list))
        
        # ---- aggcore_mate per n ----
        for m in methods:
            NMI[m].append(np.mean(score[m]))
            STD[m].append(np.std(score[m]))
            TIME[m].append(np.mean(tlist[m]))

    progress_bar.close()
    print("Processing completed.")

    # ---- keep your original return order ----
    return (
        dense_results,
        NMI["acsbm1"], NMI["acsbm3"], NMI["acsbm1_aa"], NMI["acsbm3_aa"],
        STD["acsbm1"], STD["acsbm3"], STD["acsbm1_aa"], STD["acsbm3_aa"],
        TIME["acsbm1"], TIME["acsbm3"], TIME["acsbm1_aa"], TIME["acsbm3_aa"]
    )


