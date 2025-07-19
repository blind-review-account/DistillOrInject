import torch
import warnings
import numpy as np
import networkx as nx
from collections import defaultdict
from torch_geometric.data import Data
from utils.build_ast import build_networkx_ast
from torch_geometric.utils import from_networkx

def nx_ast_and_features_to_pyg(
    G: nx.DiGraph,
    feat_matrix: torch.Tensor,
    type2idx: dict
) -> Data:
    """
    Convert a NetworkX AST `G` and its token‐level `feat_matrix` into a single PyG Data object.

    If feat_matrix is None, only positional and type features are used.
    Assumptions about G’s node attributes:
      - G.nodes[n]['token_start'] (int): index of first row in feat_matrix for this node
      - G.nodes[n]['token_end']   (int): index *after* last row in feat_matrix for this node
      - G.nodes[n]['type']        (str): node type string
      - G.nodes[n]['pos']         (torch.FloatTensor or sequence) if present
    
    Returns:
      PyG Data with:
        • data.x          → Tensor [num_nodes, feat_dim + pos_dim + 1]
        • data.edge_index → Tensor [2, num_edges]
        • data.num_nodes  → number of nodes
        • data.node_id    → list of original NetworkX node‐IDs (tuples)
    """
    feat_dim = 0
    if feat_matrix is not None:
        # 1) Ensure feat_matrix is on CPU float32
        if feat_matrix.device.type != "cpu":
            feat_matrix = feat_matrix.cpu()
        feat_matrix = feat_matrix.float()  # [num_tokens, feat_dim]
        feat_dim = feat_matrix.size(1)
    
    # 2) Determine pos_dim by inspecting any node's 'pos' attribute
    pos_dim = 0
    for _, data in G.nodes(data=True):
        if "pos" in data:
            pos_tensor = data["pos"]
            if isinstance(pos_tensor, torch.Tensor):
                pos_dim = pos_tensor.numel()
            else:
                pos_dim = len(pos_tensor)
            break
    
    # 3) Build list of nodes in deterministic order
    node_list = list(G.nodes())
    num_nodes = len(node_list)
    
    # 4) Preallocate numpy array for node features
    total_dim = feat_dim + pos_dim + 1
    node_feats_np = np.zeros((num_nodes, total_dim), dtype=np.float32)
    
    # 5) Fill feature array
    for i, node_id in enumerate(node_list):
        data = G.nodes[node_id]
        parts = []
        
        # 5a) Pool token‐level features
        if feat_dim > 0:
            start = data["token_start"]
            end   = data["token_end"]
            if end > start:
                pooled = feat_matrix[start:end + 1].mean(dim=0).numpy()  # [feat_dim]
            elif end == start: 
                pooled = feat_matrix[start].numpy()
            else:
                print("Problem")
                pooled = np.zeros(feat_dim, dtype=np.float32)
            parts.append(pooled)
        
        # 5b) Get pos vector
        if pos_dim > 0:
            pos_tensor = data["pos"]
            if isinstance(pos_tensor, torch.Tensor):
                pos_np = pos_tensor.cpu().numpy().astype(np.float32)
            else:
                pos_np = np.array(pos_tensor, dtype=np.float32)
            parts.append(pos_np)
        
        # 5c) Map node type → integer
        node_type = data.get("type", None)
        type_idx = type2idx.get(node_type, len(type2idx))
        type_feat = np.array([float(type_idx)], dtype=np.float32)
        parts.append(type_feat)
        
        # 5d) Assign features to node
        G.nodes[node_id]['x'] = np.concatenate(parts, axis=0)
    
    # 7) Convert to PyG Data
    data = from_networkx(G, group_node_attrs=["x"])
    
    # 8) Store original node IDs
    data.node_id = node_list
    
    return data


def extract_graph_and_features(
    sources,
    langs,
    tokenizer,
    model,
    max_length: int = None,
    stride: int = 32,
    device: str = None,
    clean: bool = True,
    pos_dim: int = 16,
    use_posenc: bool = False
):
    """
    Build AST graphs from source code, extract transformer-based features, and return both feature matrices and cleaned graphs.

    Args:
        sources (str or List[str]): One or more source code strings to process.
        langs (str or List[str]): Corresponding programming language(s) for each source string.
        tokenizer: Pre-loaded Hugging Face tokenizer (e.g., AutoTokenizer) already moved to the appropriate device.
        model: Pre-loaded Hugging Face model (e.g., AutoModel) already moved to `device`.
        max_length (int, optional): Maximum token length for transformer inputs. If None, uses model defaults.
        stride (int): Overlap stride when tokenizing long inputs for sliding-window feature extraction.
        device (str or torch.device, optional): Compute device for model inference ("cuda" or "cpu"). 
            If None, will default to CUDA if available, otherwise CPU.
        clean (bool): Whether to apply cleaning/preprocessing to the source code before tokenization.
        pos_dim (int): Dimensionality for any positional encoding used during AST building.
        use_posenc (bool): Whether to include positional encodings in the AST construction.

    Returns:
        Tuple[List[torch.Tensor], List[GraphType]]:
            - feature_matrices: A list of tensors containing transformer-derived features for each source.
            - cleaned_graphs: A list of cleaned/processed graph objects (e.g., NetworkX ASTs) corresponding to each source.
    """

    # If `device` wasn’t provided, choose CUDA if available, else CPU.
    if device is None:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Normalize `sources` to a list of strings
    #    so we can loop over it uniformly.
    if isinstance(sources, str):
        instances_sources = [sources]
    else:
        instances_sources = sources

    # 2. Normalize `langs` to a list as well,
    #    matching the length of `instances_sources`.
    if isinstance(langs, str):
        instances_langs = [langs]
    else:
        instances_langs = langs

    # 3. Build AST graphs for each source-language pair.
    #    `build_networkx_ast` is assumed to return a graph-like object.
    instances_asts = [
        build_networkx_ast(
            source=src,
            lang=lang,
            pos_dim=pos_dim,
            use_posenc=use_posenc
        )
        for src, lang in zip(instances_sources, instances_langs)
    ]
    if model is not None and tokenizer is not None:
        if hasattr(model.config, 'max_position_embeddings') and model.config.max_position_embeddings < 1e12:
             model_config_max_len = model.config.max_position_embeddings
        elif hasattr(model.config, 'n_positions') and model.config.n_positions < 1e12: # For GPT2-like models
             model_config_max_len = model.config.n_positions
        elif hasattr(model.config, 'seq_length') and model.config.seq_length < 1e12: # For some other models
             model_config_max_len = model.config.seq_length
        model_config_max_len = model_config_max_len - tokenizer.num_special_tokens_to_add()
        max_length = max_length if max_length is not None else model_config_max_len
        # 4. Extract transformer-based features from each source and its AST graph.
        #    `extract_tranformer_features` should leverage the pre-loaded `tokenizer` and `model`
        #    (both already on `device`) to produce feature matrices and optionally clean the graphs.
        feature_matrices, cleaned_graphs = extract_tranformer_features(
            sources=instances_sources,
            graphs=instances_asts,
            tokenizer=tokenizer,
            model=model,
            max_length=max_length,
            stride=stride,
            device=device,
            clean=clean
        )

        # 5. Return the list of feature tensors and the corresponding cleaned graphs.
        return feature_matrices, cleaned_graphs
    else : #In case there is no model and tokenizer specificied, we return the graphs 
        return [None for _ in instances_asts], instances_asts


def tokenize_with_chunking(code_snippets, tokenizer, max_length = None, stride = 32):
    """
    Tokenize one or more code snippets using a sliding-window chunking strategy.

    Each snippet is split into overlapping chunks of token IDs if it exceeds the model's max length.
    Returns a list of chunk dictionaries containing:
      - "snippet_id": index of the snippet in the original input list
      - "chunk_index": index of this chunk for that snippet
      - "input_ids": list of token IDs for the chunk
      - "offset_mapping": list of (char_start, char_end) tuples for each token

    Args:
        code_snippets (str or List[str]): Single code string or list of code strings to tokenize.
        tokenizer: Huggingface tokenizer instance (e.g., CodeBERT tokenizer).
        max_length (int, optional): Maximum number of tokens per chunk. Defaults to tokenizer.model_max_length.
        stride (int): Number of tokens to overlap between consecutive chunks.

    Returns:
        List[dict]: List of chunk metadata dictionaries.
    """
    if max_length is None :
        max_length = tokenizer.model_max_length
    
    all_chunks = [] 
    if isinstance(code_snippets, str): 
        instances = [code_snippets]
    else :
        instances = code_snippets
        
    for idx, code in enumerate(instances): 
        encoded = tokenizer(
            code,
            return_offsets_mapping=True,
            add_special_tokens=False,  # no special tokens yet for clean offsets
            max_length=max_length,
            truncation=True,
            stride=stride,
            return_overflowing_tokens=True
        )
        
        for chunk_i, (input_ids, offsets) in enumerate(zip(encoded["input_ids"], encoded["offset_mapping"])):
            all_chunks.append({
                "snippet_id": idx,
                "chunk_index": chunk_i,
                "input_ids": input_ids,
                "offset_mapping": offsets
            })
    
    return all_chunks



def get_code_feature_matrix(
    code_snippets,
    tokenizer,
    model,
    max_length=None,
    stride=32,
    device=None
):
    """
    Build a “stitched” feature matrix for each snippet by concatenating chunk outputs
    and dropping overlap. Returns:
      - feature_matrices: List[FloatTensor], each [num_tokens, hidden_dim]
      - token_index_maps:  List[dict raw_idx → final_row_idx]
        (in this version raw_idx == final_row_idx)
    """
    if isinstance(code_snippets, str):
        instances = [code_snippets]
    else:
        instances = code_snippets

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    hidden_dim = model.config.hidden_size

    if hasattr(model.config, 'max_position_embeddings') and model.config.max_position_embeddings < 1e12:
         model_config_max_len = model.config.max_position_embeddings
    elif hasattr(model.config, 'n_positions') and model.config.n_positions < 1e12: # For GPT2-like models
         model_config_max_len = model.config.n_positions
    elif hasattr(model.config, 'seq_length') and model.config.seq_length < 1e12: # For some other models
         model_config_max_len = model.config.seq_length
        
    model_config_max_len = model_config_max_len - tokenizer.num_special_tokens_to_add()
    max_length = max_length if max_length is not None else model_config_max_len
        
    # 1) Tokenize into overlapping chunks
    chunks = tokenize_with_chunking(
        code_snippets=instances,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride
    )
    # Each chunk: { "snippet_id", "chunk_index", "input_ids", "offset_mapping" }

    # 2) Build a single padded batch for all chunks
    all_ids = [torch.tensor(c["input_ids"], dtype=torch.long) for c in chunks]
    max_chunk_len = max(ids.size(0) for ids in all_ids)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    batch_ids  = torch.full((len(all_ids), max_chunk_len), pad_id, dtype=torch.long)
    batch_mask = torch.zeros((len(all_ids), max_chunk_len), dtype=torch.long)
    for i, ids in enumerate(all_ids):
        L = ids.size(0)
        batch_ids[i, :L]  = ids
        batch_mask[i, :L] = 1
    batch_ids, batch_mask = batch_ids.to(device), batch_mask.to(device)
    
    # 3) Run the model to get hidden_states
    with torch.no_grad():
        # Check if the model is an encoder-decoder architecture.
        # We want the encoder's output for feature extraction.
        if hasattr(model.config, 'is_encoder_decoder') and model.config.is_encoder_decoder:
            # For encoder-decoder models, explicitly call the encoder part.
            # T5Model (which AutoModel resolves to for CodeT5p) has an .encoder attribute.
            outputs = model.encoder(input_ids=batch_ids, attention_mask=batch_mask, return_dict=True)
        else:
            # For encoder-only models (e.g., BERT, RoBERTa),
            # the direct call to model gives the encoder's output.
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask, return_dict=True)
    hidden_states = outputs.last_hidden_state  # [num_chunks, max_chunk_len, hidden_dim]

    # 4) Group per snippet: (chunk_index, real_len, embedding_tensor)
    per_snippet = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        sid      = chunk["snippet_id"]
        cidx     = chunk["chunk_index"]
        real_len = len(chunk["input_ids"])
        emb      = hidden_states[idx]  # [max_chunk_len, hidden_dim]
        per_snippet[sid].append((cidx, real_len, emb))

    # 5) Stitch each snippet’s chunks, dropping the first 'stride' rows of each chunk after the first
    feature_matrices = []
    token_index_maps = []
    num_snippets     = len(instances)

    for sid in range(num_snippets):
        chunk_list = per_snippet[sid]
        chunk_list.sort(key=lambda x: x[0])  # sort by chunk_index

        stitched = []
        for j, (cidx, real_len, emb) in enumerate(chunk_list):
            emb = emb[:real_len]  # trim to actual token count
            if j == 0:
                stitched.append(emb.cpu())
            else:
                if real_len > stride:
                    stitched.append(emb[stride:].cpu())

        if not stitched:
            mat = torch.zeros((0, hidden_dim), dtype=torch.float32)
            feature_matrices.append(mat)
            token_index_maps.append({})
        else:
            mat = torch.cat(stitched, dim=0)  # [num_tokens, hidden_dim]
            feature_matrices.append(mat)
            token_index_maps.append({i: i for i in range(mat.size(0))})

    return feature_matrices, token_index_maps
    
def map_individual_graph_to_token_indices_dedup(
    source: str,
    G: nx.DiGraph,
    tokenizer,
    token_index_map: dict
) -> dict:
    """
    Given source + AST G + token_index_map (raw_idx→dedup_row), return
    node_id → (start_row, end_row) or None.
    """
    raw_byte_spans = get_token_byte_spans(source, tokenizer)
    node_to_span = {}

    for node_id, data in G.nodes(data=True):
        n_start, n_end = data["start_byte"], data["end_byte"]

        # Collect raw indices whose byte span lies inside [n_start,n_end]
        inside_raw = [
            i for i, (b_s, b_e) in enumerate(raw_byte_spans)
            if (b_s >= n_start) and (b_e <= n_end)
        ]
        if not inside_raw:
            node_to_span[node_id] = None
            continue

        # Map only those raw indices that survived deduplication
        final_positions = [ token_index_map[r] for r in inside_raw if r in token_index_map ]

        if not final_positions:
            node_to_span[node_id] = None
        else:
            node_to_span[node_id] = (min(final_positions), max(final_positions))

    return node_to_span


def map_nodes_to_token_indices_batch_dedup(
    sources: list,
    graphs: list,
    tokenizer,
    token_index_maps: list
) -> list:
    """
    Batch‐version: for each i, calls map_individual_graph_to_token_indices_dedup(
       sources[i], graphs[i], tokenizer, token_index_maps[i]
    ) and returns a list of those dicts.
    """
    batch_mappings = []
    if isinstance(sources, str):
        source_list = [sources]
    else:
        source_list = sources

    if isinstance(graphs, nx.DiGraph):
        graph_list = [graphs]
    else:
        graph_list = graphs

    for i, G in enumerate(graph_list):
        mapping = map_individual_graph_to_token_indices_dedup(
            source_list[i],
            G,
            tokenizer,
            token_index_maps[i]
        )
        batch_mappings.append(mapping)

    return batch_mappings

def extract_transformer_features(
    sources,
    graphs,
    tokenizer,
    model,
    max_length: int = None,
    stride: int = 32,
    device: str = None,
    clean: bool = True
):
    """
    Returns:
      - feature_matrices: List[Tensor]   # each [num_dedup_tokens, hidden_dim]
      - cleaned_graphs:  List[nx.DiGraph] where each node has valid token_start/token_end
    """
    # 1) Normalize to lists
    if isinstance(sources, str):
        source_list = [sources]
    else:
        source_list = sources

    if isinstance(graphs, nx.DiGraph):
        graph_list = [graphs]
    else:
        graph_list = graphs

    if len(source_list) != len(graph_list):
        raise ValueError("`sources` and `graphs` must have same length.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    if hasattr(model.config, 'max_position_embeddings') and model.config.max_position_embeddings < 1e12:
         model_config_max_len = model.config.max_position_embeddings
    elif hasattr(model.config, 'n_positions') and model.config.n_positions < 1e12: # For GPT2-like models
         model_config_max_len = model.config.n_positions
    elif hasattr(model.config, 'seq_length') and model.config.seq_length < 1e12: # For some other models
         model_config_max_len = model.config.seq_length
    model_config_max_len = model_config_max_len - tokenizer.num_special_tokens_to_add()
    max_length = max_length if max_length is not None else model_config_max_len
    
    # 2) Compute deduped feature_matrices + token_index_maps in one shot
    feature_matrices, token_index_maps = get_code_feature_matrix(
        code_snippets=source_list,
        tokenizer=tokenizer,
        model=model,
        max_length=max_length,
        stride=stride,
        device=device
    )

    # 3) Build node‐to‐dedup span mappings
    mappings = map_nodes_to_token_indices_batch_dedup(
        sources=source_list,
        graphs=graph_list,
        tokenizer=tokenizer,
        token_index_maps=token_index_maps
    )

    # 4) Attach spans (and prune if clean=True)
    for G, node_to_span in zip(graph_list, mappings):
        attach_spans_and_clean_individual_graph(G, node_to_span, clean=clean)

    return feature_matrices, graph_list
    
def char_to_byte_map(source: str):
    """
    Build a lookup table that maps each character index to its cumulative byte offset
    in the UTF-8 encoding of `source`. This helps convert (char_start, char_end) → (byte_start, byte_end).

    Returns:
        List[int]: List of length len(source)+1, where index i stores byte-offset of source[:i].
    """
    # Build a simple char→byte lookup table
    char_to_byte = [0] * (len(source) + 1)
    for i in range(1, len(source) + 1):
        char_to_byte[i] = len(source[:i].encode("utf-8"))
    return char_to_byte

def get_token_byte_spans(source: str, tokenizer):
    """
    Tokenize `source` without adding special tokens, retrieve character‐level offsets,
    then convert those to byte‐level spans with char_to_byte_map. 
    This version avoids the “>512 tokens” warning by temporarily raising model_max_length.
    """

    # 1) Temporarily set model_max_length high enough so no warning is triggered
    old_max = tokenizer.model_max_length
    tokenizer.model_max_length = 10**6   # or len(source) + 1, etc.

    # 2) Tokenize normally (no truncation; we want ALL offsets)
    encoding = tokenizer(
        source,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False
    )

    # 3) Restore the original model_max_length
    tokenizer.model_max_length = old_max

    # 4) Convert char offsets to byte offsets
    offsets = encoding["offset_mapping"]
    c2b = char_to_byte_map(source)

    token_byte_spans = []
    for (char_s, char_e) in offsets:
        byte_s = c2b[char_s]
        byte_e = c2b[char_e]
        token_byte_spans.append((byte_s, byte_e))
    return token_byte_spans

        
def map_individual_graph_to_token_indices(
    source: str,
    G: nx.DiGraph,
    tokenizer,
    token_index_map: dict
) -> dict:
    """
    Map each AST node in G to (start_idx, end_idx) in the *deduplicated* feature matrix,
    using `token_index_map` { raw_idx → final_row_idx }.

    If none of the raw tokens under a node survive deduplication, return None for that node.

    Returns:
      node_to_span: { node_id: (start_final, end_final) or None }
    """
    # 1) Get the full raw token -> byte spans for this snippet
    raw_byte_spans = get_token_byte_spans(source, tokenizer)
    node_to_span = {}

    for node_id, data in G.nodes(data=True):
        n_start = data["start_byte"]
        n_end   = data["end_byte"]

        # 2) Collect raw indices lying inside the node's byte span
        inside_raw = [
            i for i, (b_s, b_e) in enumerate(raw_byte_spans)
            if b_s >= n_start and b_e <= n_end
        ]
        if not inside_raw:
            node_to_span[node_id] = None
            continue

        # 3) Among those raw indices, keep only those present in token_index_map
        final_positions = [ token_index_map[r] for r in inside_raw if r in token_index_map ]
        if not final_positions:
            node_to_span[node_id] = None
        else:
            start_final = min(final_positions)
            end_final   = max(final_positions)
            node_to_span[node_id] = (start_final, end_final)

    return node_to_span
    

def map_nodes_to_token_indices_batch(
    sources: list,
    graphs: list,
    tokenizer
) -> list:
    """
    Batch version of `map_nodes_to_token_indices`. Takes a list of source strings
    and a parallel list of AST graphs, returning a list of mappings.

    For each index i, calls `map_nodes_to_token_indices(sources[i], graphs[i], tokenizer)`
    and collects the resulting dict into a list.

    Args:
        sources (List[str]): List of raw source code strings.
        graphs (List[nx.DiGraph]): List of AST graphs; graphs[i] corresponds to sources[i].
        tokenizer: Huggingface tokenizer instance.

    Returns:
        List[dict]: A list of length N, where each element is the node-to-token mapping
                    dict for the corresponding (source, graph) pair.
    """
    batch_mappings = []
    if isinstance(sources, str):
        source_instances = [sources]
    else : 
        source_instances = sources
        
    if isinstance(graphs, nx.DiGraph):
        source_graphs = [graphs]
    else :
        source_graphs = graphs
        
    for src, G in zip(source_instances, source_graphs):
        mapping = map_individual_graph_to_token_indices(src, G, tokenizer)
        batch_mappings.append(mapping)
    return batch_mappings


def clean_graph(G: nx.DiGraph, node_to_tokens: dict):
    """
    Remove nodes without any token span from the AST graph, reconnecting their children to the grandparent.

    For each node_id in G:
      - If node_to_tokens[node_id] is None and node is not the root (path=()):
          1. Let parent_id = tuple(path[:-1])
          2. For each child of node, add an edge (parent_id -> child)
          3. Remove node_id from G

    This ensures that nodes covering no tokens are pruned, and their children
    are reattached to maintain a valid tree structure.

    Args:
        G (nx.DiGraph): AST graph with node attributes including "path".
        node_to_tokens (dict): Mapping from node_id to (start_idx, end_idx) or None.
    """
    # Identify all removable nodes: those with no spans and not the root
    nodes_to_remove = [nid for nid, span in node_to_tokens.items() if span is None and nid != ()]

    # Rewire children and remove nodes
    for nid in nodes_to_remove:
        # Determine parent_id from the node's path
        parent_path = G.nodes[nid]["path"][:-1]
        parent_id = tuple(parent_path)
        
        # Reconnect each child of nid to parent_id (skip if parent is missing)
        for child in list(G.successors(nid)):
            if parent_id in G:
                G.add_edge(parent_id, child)
            # else: if parent is rootless or missing, skip reconnect

        # Remove the node from the graph
        G.remove_node(nid)
        
def attach_spans_and_clean_individual_graph(G: nx.DiGraph, node_to_tokens: dict, clean: bool = True):
    """
    Attach token-span indices to AST nodes in a single graph, and optionally prune nodes with no spans.

    For each node_id in G:
      - If node_to_tokens[node_id] is not None, record its token range by setting:
            G.nodes[node_id]["token_start"] = start_idx
            G.nodes[node_id]["token_end"]   = end_idx
      - If node_to_tokens[node_id] is None and node is not the root (node_id == ()), mark it for removal.

    If clean=True, calls clean_graph(G, node_to_tokens) after all spans are attached. That function:
      1. Reconnects children of any removable node to that node’s parent.
      2. Deletes the removable node from G.

    Args:
        G (nx.DiGraph): AST graph where each node has a "path" attribute (list of ints).
                        Nodes should also have "start_byte" and "end_byte".
        node_to_tokens (dict): Mapping from node_id to (start_idx, end_idx) or None.
        clean (bool): If True, prune nodes with no spans by reconnecting children to their parent.
    """
    # 1) Attach token_start/token_end for nodes that have a valid span
    #    and collect nodes without spans (excluding the root) for potential removal.
    nodes_to_remove = []
    for node_id, span in node_to_tokens.items():
        if span is None and node_id != ():
            # Node has no tokens and is not the root ⇒ schedule for removal
            nodes_to_remove.append(node_id)
        elif span is not None and node_id in G:
            # Unpack the token indices and attach as attributes
            start_idx, end_idx = span
            G.nodes[node_id]["token_start"] = start_idx
            G.nodes[node_id]["token_end"]   = end_idx

    # 2) If requested, prune nodes with no spans:
    if clean:
        clean_graph(G, node_to_tokens)


def attach_spans_and_clean(
    graphs: list,
    mappings: list,
    clean: bool = False
):
    """
    Batch‐process a list of AST graphs and their corresponding node-to-token mappings:
      1. Ensures both graphs and mappings are treated as lists.
      2. For each pair (graph, node_to_tokens), calls attach_spans_and_clean_individual_graph,
         which attaches token_start/token_end and optionally prunes nodes with no spans.

    Args:
        graphs (nx.DiGraph or List[nx.DiGraph]): Single AST graph or list of AST graphs.
        mappings (dict or List[dict]): Single node_to_tokens dict or list of such dicts.
                                        Each dict maps node_id → (start_idx, end_idx) or None.
        clean (bool): If True, prune nodes with no spans in each graph.

    Notes:
        - If a single graph is passed, it is wrapped into a one-element list.
        - If a single mapping is passed, it is wrapped into a one-element list.
        - `clean` is forwarded to attach_spans_and_clean_individual_graph for each graph.
    """
    # 1) Normalize `graphs` to a list
    if isinstance(graphs, nx.DiGraph):
        source_graphs = [graphs]
    else:
        source_graphs = graphs

    # 2) Normalize `mappings` to a list
    if isinstance(mappings, dict):
        source_mappings = [mappings]
    else:
        source_mappings = mappings

    # 3) Iterate in parallel, attaching spans and optionally cleaning each graph
    for g, mapping in zip(source_graphs, source_mappings):
        attach_spans_and_clean_individual_graph(g, mapping, clean)
        
        
def extract_tranformer_features(
    sources,
    graphs,
    tokenizer,
    model,
    max_length: int = None,
    stride: int = 32,
    device: str = None,
    clean: bool = True
):
    """
    Build feature_matrices and cleaned_graphs so that every node's token_start/end
    is valid under the deduplicated feature matrix.
    """
    # 1) Normalize inputs to lists
    if isinstance(sources, str):
        source_list = [sources]
    else:
        source_list = sources

    if isinstance(graphs, nx.DiGraph):
        graph_list = [graphs]
    else:
        graph_list = graphs

    if len(source_list) != len(graph_list):
        raise ValueError("sources and graphs must match length.")

    # 2) Choose device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    
    if hasattr(model.config, 'max_position_embeddings') and model.config.max_position_embeddings < 1e12:
         model_config_max_len = model.config.max_position_embeddings
    elif hasattr(model.config, 'n_positions') and model.config.n_positions < 1e12: # For GPT2-like models
         model_config_max_len = model.config.n_positions
    elif hasattr(model.config, 'seq_length') and model.config.seq_length < 1e12: # For some other models
         model_config_max_len = model.config.seq_length
    model_config_max_len = model_config_max_len - tokenizer.num_special_tokens_to_add()
    max_length = max_length if max_length is not None else model_config_max_len
    # 3) Compute deduplicated features + token_index_maps
    feature_matrices, token_index_maps = get_code_feature_matrix(
        code_snippets=source_list,
        tokenizer=tokenizer,
        model=model,
        max_length=max_length,
        stride=stride,
        device=device
    )
    #    feature_matrices[i]: Tensor [num_dedup_tokens_i, hidden_dim]
    #    token_index_maps[i]:  dict { raw_idx → final_row_idx }

    # 4) For each (source, graph, token_index_map), compute dedup‐aware node spans
    mappings = map_nodes_to_token_indices_batch_dedup(
        sources=source_list,
        graphs=graph_list,
        tokenizer=tokenizer,
        token_index_maps=token_index_maps
    )

    # 5) Attach spans & prune if requested
    for G, node_to_span in zip(graph_list, mappings):
        attach_spans_and_clean_individual_graph(G, node_to_span, clean=clean)

    return feature_matrices, graph_list