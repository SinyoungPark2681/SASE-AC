from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse, stats

from . import models2


class NetworkTooSmallError(RuntimeError):
    pass


@dataclass
class GeneratedNetwork:
    A: Any # scipy.sparse
    theta: np.ndarray
    Z: np.ndarray
    Z_tilde: np.ndarray
    block_sizes: list[int]
    model: models2.MultiCovariateModel

    @property
    def n_blocks(self) -> int:
        return len(self.block_sizes)

    @property
    def n_nodes(self) -> int:
        return self.A.shape[0]
    
    @property
    def n_edges(self) -> int:
        return self.A.count_nonzero()


def generate_sparse_block(size, prob, symmetric=False, random_state=None):
    """
    Generates a random block of binary entries where each entry is 1 w.p. prob
    If symmetric=True, returns a symmetric block with a zero on the diagonal.
    """
    density = stats.binom.rvs(size[0] * size[1], prob, size=1, random_state=random_state).item() / (size[0] * size[1])
    m = sparse.random(size[0], size[1], density, random_state=random_state)
    m.data[:] = 1
    
    if symmetric:
        if size[0] != size[1]:
            raise RuntimeError("symmetric matrix must be square")
        m = sparse.triu(m, k=1) + sparse.triu(m, k=1).transpose()
    
    return m

def generate_sparse_sbm(block_sizes, block_probs, random_state=None):
    """
    Generate a stochastic block model using fixed block sizes and connectivity matrix
    """
    k = len(block_sizes)
    blocks = [[None for i in range(k)] for j in range(k)]
    
    for i in range(k):
        for j in range(i, k):
            blocks[i][j] = generate_sparse_block(
                (block_sizes[i], block_sizes[j]),
                block_probs[i,j],
                symmetric=(i == j),
                random_state=random_state
            )
            if i < j:
                blocks[j][i] = blocks[i][j].transpose()
    
    return sparse.bmat(blocks)

def generate_sparse_sbm_directed(block_sizes, block_probs, random_state=None):
    k = len(block_sizes)
    blocks = [[None for _ in range(k)] for _ in range(k)]
    
    for i in range(k):
        for j in range(k):
            blocks[i][j] = generate_sparse_block(
                (block_sizes[i], block_sizes[j]),
                block_probs[i, j],
                symmetric=False,        # <-- Important: always False
                random_state=random_state
            )
    
    return sparse.bmat(blocks)


def generate_network(model: models2.MultiCovariateModel, ndd: models2.NodeDataDistribution, n: int, random_state: np.random.RandomState = None) -> GeneratedNetwork:
    theta, Z = ndd.draw(n, random_state=random_state)
    Z_tilde = model.flatten_Z(Z)
    L_tilde = np.prod([c.n_levels for c in model.covariates])
    theta_tilde = L_tilde * theta + Z_tilde # same as applying tuple_id(...) over rows
    counts = Counter(theta_tilde)
    block_sizes = [counts[i] for i in range(model.n_communities * L_tilde)]

    # This could probably be handled more gracefully...
    if len(counts.keys()) < model.n_communities * L_tilde:
        raise NetworkTooSmallError("Generated network does not have a node of every type. Consider using a larger n.")

    return GeneratedNetwork(
        A=generate_sparse_sbm(block_sizes, model.B_tilde(), random_state=random_state),
        theta=theta[np.argsort(theta_tilde)],
        Z=Z[np.argsort(theta_tilde), :],
        Z_tilde=Z_tilde[np.argsort(theta_tilde)],
        block_sizes=block_sizes,
        model=model
    )

def generate_network_directed(model, ndd, n, directed=False, random_state=None):

    if isinstance(random_state, (int, np.integer)):
        random_state = np.random.default_rng(random_state)

    theta, Z = ndd.draw(n, random_state=random_state)
    Z_tilde = model.flatten_Z(Z)

    L_tilde = np.prod([c.n_levels for c in model.covariates])
    theta_tilde = L_tilde * theta + Z_tilde
    counts = Counter(theta_tilde)
    block_sizes = [counts[i] for i in range(model.n_communities * L_tilde)]

    if len(counts.keys()) < model.n_communities * L_tilde:
        raise NetworkTooSmallError("Generated network does not have a node of every type.")

    if directed:
        A = generate_sparse_sbm_directed(block_sizes, model.B_tilde(), random_state=random_state)
    else:
        A = generate_sparse_sbm(block_sizes, model.B_tilde(), random_state=random_state)

    return GeneratedNetwork(
        A=A,
        theta=theta[np.argsort(theta_tilde)],
        Z=Z[np.argsort(theta_tilde), :],
        Z_tilde=Z_tilde[np.argsort(theta_tilde)],
        block_sizes=block_sizes,
        model=model
    )

def generate_network_with_custom_data(model, theta, Z, random_state=None):
    # theta: array of shape (n,)  -> community labels
    # Z: array of shape (n, p)    -> covariates (each Z[:,m] ∈ {0,1})

    n = len(theta)
    Z_tilde = model.flatten_Z(Z)

    L_tilde = np.prod([c.n_levels for c in model.covariates])
    theta_tilde = L_tilde * theta + Z_tilde

    from collections import Counter
    counts = Counter(theta_tilde)
    block_sizes = [counts[i] for i in range(model.n_communities * L_tilde)]

    A = generation.generate_sparse_sbm(block_sizes, model.B_tilde(), random_state=random_state)

    return generation.GeneratedNetwork(
        A=A,
        theta=theta,
        Z=Z,
        Z_tilde=Z_tilde,
        block_sizes=block_sizes,
        model=model
    )
