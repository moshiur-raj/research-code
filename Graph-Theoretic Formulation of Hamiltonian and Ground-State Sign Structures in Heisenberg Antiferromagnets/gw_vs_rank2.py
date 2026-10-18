from math import ceil, sqrt
import jax
import optax

import numpy as np
import jax.numpy as jnp
import scipy as sp
import netket as nk
from netket.operator.spin import sigmax as sx, sigmay as sy, sigmaz as sz
from tqdm import tqdm
import pandas as pd

from gw_maxcut import gw_maxcut
from rank2_maxcut import rank2_maxcut

def gen_adj(hi, g):
    ham_pm_1 = nk.operator.LocalOperator(hilbert=hi, dtype=jnp.complex128)
    ham_pm_2 = nk.operator.LocalOperator(hilbert=hi, dtype=jnp.complex128)
    for (i, j, c) in g.edges(return_color=True):
        if c == 0:
            ham_pm_1 += sx(hi, i) @ sx(hi, j) + sy(hi, i) @ sy(hi, j)
        else:
            ham_pm_2 += sx(hi, i) @ sx(hi, j) + sy(hi, i) @ sy(hi, j)

    adj1 = ham_pm_1.to_sparse().astype(bool).astype(float)
    adj2 = ham_pm_2.to_sparse().astype(bool).astype(float)

    return sp.sparse.tril(adj1).tocsr(), sp.sparse.tril(adj2).tocsr()


def gen_weight_matrix(psi, adj_1, adj_2, coups):
    j1, j2 = coups
    diag = sp.sparse.diags(psi)
    w1 = j1 * diag @ adj_1 @ diag
    w2 = j2 * diag @ adj_2 @ diag

    w = w1 + w2
    w.eliminate_zeros()

    return w.tocoo()

labels = ['Chain, 16', 'Square, 4x4', 'Triangular, 4x4']
graphs = [nk.graph.Chain (16, pbc=True, max_neighbor_order=2),
          nk.graph.Square(4,  pbc=True, max_neighbor_order=2),
          nk.graph.Triangular((4, 4),  pbc=True, max_neighbor_order=2)]

optimizer = optax.adam(1, eps=1e-14)
n_iter = 500
n_hyperplanes = 128

for label, g in zip(labels, graphs):
    hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes, total_sz=0)
    adj1, adj2 = gen_adj(hi, g)

    ham1 = nk.operator.Heisenberg(
            hi, g, J=[1, 0], sign_rule=[False, False]
            ).to_sparse().real.tocsr()
    ham2 = nk.operator.Heisenberg(
            hi, g, J=[0, 1], sign_rule=[False, False]
            ).to_sparse().real.tocsr()

    dj2 = .1
    j2_list = np.arange(0, 1 + dj2/2, dj2)

    sign_match_gw = []
    sign_match_rank2 = []
    n = hi.n_states
    # k = ceil(sqrt(2 * n))
    k=2
    for j2 in tqdm(j2_list):
        ham = ham1 + j2 * ham2
        energy, psi = sp.sparse.linalg.eigsh(ham, k=1, which='SA')
        psi = psi.squeeze()

        psi_abs = np.abs(psi)
        cond = psi_abs < psi_abs.max() * 1e-9
        psi[cond] = 0
        psi_abs[cond] = 0
        signs = np.sign(psi)

        w = gen_weight_matrix(psi_abs, adj1, adj2, coups=[1, j2])

        signs_gw = gw_maxcut(w.data, w.row, w.col,
                             n=n, k=k, n_hyperplanes=n_hyperplanes, n_iter=n_iter,
                             optimizer=optimizer).block_until_ready()
        signs_rank2 = rank2_maxcut(w.data, w.row, w.col,
                                   n=n, n_iter=n_iter, optimizer=optimizer).block_until_ready()

        num_match = jnp.maximum(
                jnp.count_nonzero(signs_gw == signs),
                jnp.count_nonzero(signs_gw == -signs),
                )
        sign_match_gw.append(float(num_match / jnp.count_nonzero(signs)))

        num_match = jnp.maximum(
                jnp.count_nonzero(signs_rank2 == signs),
                jnp.count_nonzero(signs_rank2 == -signs),
                )
        sign_match_rank2.append(float(num_match / jnp.count_nonzero(signs)))
    
    df = pd.DataFrame({
        'J2': j2_list.tolist(),
        'GW': sign_match_gw,
        'Rank2': sign_match_rank2
        })

    print(df.round(3).T)
