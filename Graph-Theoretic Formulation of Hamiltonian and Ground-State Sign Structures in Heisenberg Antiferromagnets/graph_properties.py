import jax
import jax.numpy as jnp
import scipy as sp
import netket as nk
from netket.operator.spin import sigmax as sx, sigmay as sy, sigmaz as sz
import pandas as pd
from tqdm import tqdm

jax.config.update('jax_enable_x64', True)

def gen_adj(g: nk.graph.Lattice) -> sp.sparse.spmatrix:
    hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes, total_sz=total_sz)

    edges = g.edges(return_color=True)
    edges_nn = [(i, j) for i, j, color in edges if color == 0]
    edges_nnn = [(i, j) for i, j, color in edges if color == 1]

    ham_pm_nn = sum([ sx(hi, i)@sx(hi, j) + sy(hi, i)@sy(hi, j) for i, j in edges_nn ])
    ham_pm_nnn = sum([ sx(hi, i)@sx(hi, j) + sy(hi, i)@sy(hi, j) for i, j in edges_nnn ])

    # 64 bit int necessary for calculating matrix powers, otherwise it overflows.
    adj_nn = ham_pm_nn.to_sparse().astype(bool).astype(jnp.int64).tocsr()
    adj_nnn = ham_pm_nnn.to_sparse().astype(bool).astype(jnp.int64).tocsr()
    adj_total = adj_nn + adj_nnn

    assert adj_total.max() == 1

    return adj_nn, adj_nnn, adj_total

def bipartivity(adj: sp.sparse.sparray) -> float:
    eigenvalues = jnp.linalg.eigvalsh(adj.todense())
    a = jnp.exp(eigenvalues).sum()
    b = jnp.exp(-eigenvalues).sum()
    beta = 1/2 + b/(2*a)

    return beta.item()

def num_triangles(adj: sp.sparse.sparray) -> int:
    return sp.sparse.linalg.matrix_power(adj, 3).trace() // 6

total_sz = 0

graphs = [
        nk.graph.Chain(length=16, pbc=True, max_neighbor_order=2),
        nk.graph.Chain(length=16, pbc=False, max_neighbor_order=2),
        nk.graph.Grid((4, 4), pbc=True, max_neighbor_order=2),
        nk.graph.Grid((4, 4), pbc=False, max_neighbor_order=2),
        nk.graph.Triangular(extent=(4, 4), pbc=True, max_neighbor_order=2),
        nk.graph.Triangular(extent=(4, 4), pbc=False, max_neighbor_order=2),
        ]

labels = [
        "1D (PBC)",
        "1D (OBC)",
        "Square (PBC)",
        "Square (OBC)",
        "Triangular (PBC)",
        "Triangular (OBC)",
        ]

columns = [
        "Lattice", "Vertices",
        "Edges (NN)", "Triangles (NN)", "Beta (NN)",
        "Edges (NNN)", "Triangles (NNN)", "Beta (NNN)",
        "Edges (Total)", "Triangles (Total)", "Beta (Total)",
        ]

data = []
for g in tqdm(graphs):
    adj_nn, adj_nnn, adj_total = gen_adj(g)

    data.append([
        adj_nn.shape[0],
        adj_nn.nnz // 2, num_triangles(adj_nn), bipartivity(adj_nn),
        adj_nnn.nnz // 2, num_triangles(adj_nnn), bipartivity(adj_nnn),
        adj_total.nnz // 2, num_triangles(adj_total), bipartivity(adj_total),
        ])

data_pd = [[i, *j] for i, j in zip(labels, data)]
df = pd.DataFrame(data_pd, columns=columns)

df.to_csv('graph_properties_unweighted.csv', index=False)

pd.set_option('display.width', 2000)
print()
print(df)
