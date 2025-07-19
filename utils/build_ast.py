import code_ast
import networkx as nx
import warnings
import torch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_networkx_ast(
    source: str,
    lang: str = "python",
    use_posenc: bool = False,
    pos_dim: int = 16
) -> nx.DiGraph:
    """
    Parse `source` code into a Tree-sitter CST and convert it into a NetworkX directed graph (DiGraph),
    with optional Laplacian‐Eigenvector positional encodings for each node.

    Each node in the resulting graph has attributes:
      - "type": CST node type (e.g., "function_definition", "identifier", etc.)
      - "start_byte": starting byte offset in `source`
      - "end_byte": ending byte offset in `source`
      - "path": list of child indices from the root to this node
      - "pos": torch.FloatTensor of length `pos_dim` (always present, zero if `use_posenc=False`)

    Edges in the graph go from a parent node to each child node, preserving the tree structure.

    Args:
        source (str):      Raw source code to parse.
        lang (str):        Language name for Tree-sitter (default: "python").
        use_posenc (bool): If True, compute and attach spectral positional encodings.
        pos_dim (int):     Number of spectral dimensions (k) to keep if `use_posenc`.
                           If `use_posenc=False`, each node’s "pos" is a zero vector of length `pos_dim`.

    Returns:
        nx.DiGraph: A directed graph where every node contains "type", "start_byte", "end_byte", "path", and "pos".
    """
    # 1) Parse into a Tree-sitter CST
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        tree = code_ast.ast(source, lang=lang, syntax_error = "ignore")
        root = tree.root_node()
    # 2) Initialize empty directed graph
    G = nx.DiGraph()

    def traverse(node, path):
        """
        Recursively walk the CST, adding each node to G with its metadata.
        """
        node_id = tuple(path)
        G.add_node(
            node_id,
            type=node.type,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            path=path.copy()
        )
        if path:
            parent_id = tuple(path[:-1])
            G.add_edge(parent_id, node_id)
        for idx, child in enumerate(node.children):
            traverse(child, path + [idx])

    traverse(root, [])

    # 3) Optionally compute and attach Laplacian‐Eigenvector PE
    if use_posenc:
        pe_tensor, node_order = compute_laplacian_pe(G, k=pos_dim)
        for i, node_id in enumerate(node_order):
            G.nodes[node_id]["pos"] = pe_tensor[i]  # shape (pos_dim,)
    else:
        zero_vec = torch.zeros(pos_dim, dtype=torch.float32)
        for node_id in G.nodes():
            G.nodes[node_id]["pos"] = zero_vec  # default zero padding
    return G


def compute_laplacian_pe(G: nx.DiGraph, k: int = 16, device = "cuda"):
    """
    Compute the first k nontrivial Laplacian eigenvectors for a given AST graph.

    Args:
        G (nx.DiGraph): Directed graph (AST) where nodes already exist.
        k (int): Number of nontrivial eigenvectors to return. (Total spectral dim = k.)

    Returns:
        pe_tensor (torch.FloatTensor): Shape (N, k), where N = number of nodes in G.
        node_order (list): List of node IDs in the same order as rows of `pe_tensor`.
                           So `pe_tensor[i]` is the k-dimensional PE for `node_order[i]`.
    """
    if device == "cuda" :
        return compute_laplacian_pe_gpu(G = G, k = k, device='cuda')
    # 1) Convert to undirected, since spectral PE ignores edge direction
    U = nx.Graph(G)

    # 2) Fix a consistent ordering of all nodes
    node_order = sorted(U.nodes())
    N = len(node_order)
    idx_map = {node_id: i for i, node_id in enumerate(node_order)}

    # 3) Build adjacency matrix A in COO format (shape N×N)
    rows, cols, data = [], [], []
    for u, v in U.edges():
        i, j = idx_map[u], idx_map[v]
        # Add both (i,j) and (j,i) to make A symmetric
        rows += [i, j]
        cols += [j, i]
        data += [1.0, 1.0]
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N))

    # 4) Compute degree matrix D and Laplacian L = D - A
    degs = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degs)
    L = D - A

    # 5) Cap effective k to at most N-1
    max_nontrivial = N - 1
    k_eff = min(k, max_nontrivial)
    
    # 6) Solve for the smallest (k+1) eigenpairs: L u = λ u
    try:
        if N <= 2000:
            L_dense = L.toarray()
            eigvals, eigvecs = np.linalg.eigh(L_dense)
            eigvecs = eigvecs[:, : (k_eff + 1)]
        else:
            eigvals, eigvecs = spla.eigsh(L, k=k_eff+1, which="SM", tol=1e-4)
    except Exception:
        # Fallback to dense if sparse solver fails
        L_dense = L.toarray()
        eigvals, eigvecs = np.linalg.eigh(L_dense)
        eigvecs = eigvecs[:, : (k_eff + 1)]

    # 7) Discard the trivial eigenvector at index 0 (constant mode), keep next k
    pe = eigvecs[:, 1 : k_eff + 1]  # shape: (N, k)
    
    # 8) Padding if necessary
    if pe.shape[1] < k:
        pad_width = k - k_eff
        pad = np.zeros((N, pad_width), dtype=pe.dtype)
        pe = np.hstack([pe, pad])
        
    # 9) Convert to torch.FloatTensor and return
    pe_tensor = torch.from_numpy(pe.astype(np.float32))  # (N, k)
    return pe_tensor, node_order

def compute_laplacian_pe_gpu(G: nx.DiGraph, k: int = 16, device='cuda'):
    """
    Compute the first k nontrivial Laplacian eigenvectors for a given AST graph,
    using GPU-accelerated dense eigendecomposition for speed.
    """
    # 1) Convert to undirected and order nodes
    U = nx.Graph(G)
    node_order = sorted(U.nodes())
    N = len(node_order)
    idx_map = {n: i for i, n in enumerate(node_order)}

    # 2) Build sparse Laplacian L
    rows, cols, data = [], [], []
    for u, v in U.edges():
        i, j = idx_map[u], idx_map[v]
        rows += [i, j]
        cols += [j, i]
        data += [1.0, 1.0]
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    degs = np.array(A.sum(axis=1)).flatten()
    L = sp.diags(degs) - A  # scipy sparse

    # 3) Convert to dense torch tensor on GPU
    L_dense = torch.from_numpy(L.toarray().astype(np.float32)).to(device)

    # 4) Compute eigenpairs on GPU
    eigvals, eigvecs = torch.linalg.eigh(L_dense)  # symmetric eigendecomposition

    # 5) Discard trivial eigenvector and select top k
    k_eff = min(k, N - 1)
    pe = eigvecs[:, 1:k_eff+1].cpu()

    # 6) Pad if needed
    if pe.shape[1] < k:
        pad = torch.zeros((N, k - k_eff), dtype=pe.dtype)
        pe = torch.cat([pe, pad], dim=1)

    del L, L_dense, eigvals, eigvecs
    torch.cuda.empty_cache()
    return pe, node_order
