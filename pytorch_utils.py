

import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import torch
import torch_geometric as pyg
from torch_geometric.data import Data, DataLoader

def sort_tree(tree, sort_prop=0):
    def _sort_tree(tree, root_index=0):
        """ Sort the tree in a depth-first order """
        ordered_index = [root_index, ]
        prog_indices = tree.edge_index[1, tree.edge_index[0] == root_index]
        if len(prog_indices) == 0:
            return ordered_index
        prog_indices = prog_indices[
            torch.argsort(tree.x[prog_indices, sort_prop], descending=True)]
        for prog_index in prog_indices:
            ordered_index.extend(_sort_tree(tree, prog_index))
        return ordered_index

    ordered_index = _sort_tree(tree, torch.tensor(0, device=tree.x.device))
    ordered_index = torch.tensor(ordered_index, device=tree.x.device)
    ordered_index_list = ordered_index.tolist()

    new_tree = tree.clone()
    for key, value in tree.items():
        if key not in ['edge_index']:
            new_tree[key] = value[ordered_index]
    new_tree.edge_index = torch.stack([
        torch.tensor([ordered_index_list.index(i.item()) for i in tree.edge_index[0]]),
        torch.tensor([ordered_index_list.index(i.item()) for i in tree.edge_index[1]])
    ])
    perm = torch.argsort(new_tree.edge_index[1])
    new_tree.edge_index = new_tree.edge_index[:, perm]

    return new_tree

def remove_nodes(tree, mass_cut, concentration_cut=0, max_num_prog=10000):
    mass_cut = torch.tensor(mass_cut, dtype=tree.x.dtype, device=tree.x.device)
    def _traverse_remove(index):
        indices = [index, ]
        prog_index = tree.edge_index[1][tree.edge_index[0] == index]
        for prog in prog_index[:max_num_prog]:
            if tree.x[prog, 0] < mass_cut:
                continue
            if tree.x[prog, 1] < concentration_cut:
                continue
            indices.extend(_traverse_remove(prog.item()))
        return indices
    indices = _traverse_remove(0)
    indices = torch.tensor(indices, device=tree.x.device)
    indices_list = indices.tolist()

    new_tree = tree.clone()
    for key, value in tree.items():
        if key not in ['edge_index']:
            new_tree[key] = value[indices]
    edge_index = [[], []]
    for i in range(len(tree.edge_index[0])):
        e1 = tree.edge_index[0, i].item()
        e2 = tree.edge_index[1, i].item()
        if e1 not in indices_list or e2 not in indices_list:
            continue
        edge_index[0].append(indices_list.index(e1))
        edge_index[1].append(indices_list.index(e2))
    new_tree.edge_index = torch.tensor(
        edge_index, device=tree.edge_index.device, dtype=torch.long)
    perm = torch.argsort(new_tree.edge_index[1])
    new_tree.edge_index = new_tree.edge_index[:, perm]
    return new_tree

def satgen_to_pyg(data, sort=False) -> Data:
    """
    Convert SatGen merger trees to PyTorch Geometric Data object.
    Only includes mass, concentration, and redshift features.

    TreeGen structure:
    - mass, concentration, parent_id: (num_branches, num_redshifts) with -99 padding
    - parent_id pattern: e.g. [-99, -99, 1, 1, 1, -99] means branch starts at index 2,
      has length 3, and parent is branch 1
    - redshift: (num_redshifts,) - time grid

    Args:
        data: dictionary containing the SatGen data

    Returns:
        PyTorch Geometric Data object with exactly N-1 edges for N nodes
    """
    # Load NPZ data
    mass = data['mass']
    cvir = data['concentration']
    parent_br_index = data['ParentID']
    redshift = data['redshift']

    # Create node features
    valid_mask = mass > 0
    valid_indices = np.where(valid_mask)

    branch_indices = valid_indices[0]
    z_indices = valid_indices[1]

    log_mass = np.log10(mass[valid_mask])
    log_cvir = np.log10(cvir[valid_mask])
    z_values = redshift[z_indices]
    node_features = np.column_stack([log_mass, log_cvir, z_values])
    x = torch.tensor(node_features, dtype=torch.float)

    # Create mapping efficiently
    node_mapping = {(b, z): i for i, (b, z) in enumerate(zip(branch_indices, z_indices))}

    # Create edges for each branch
    edges1, edges2 = [], []
    bad_halo_idx = []
    for br_index, br_parent in enumerate(parent_br_index):
        desc_z_ids = np.where(br_parent != -99)[0]
        desc_z_id = desc_z_ids[0]  # Branch start redshift
        num_halo_br = len(desc_z_ids)

        first_halo_index = node_mapping.get((br_index, desc_z_id))
        halo_indices = first_halo_index + np.arange(num_halo_br)

        # Merger edge: connect to parent branch (skip main branch)
        if br_index > 0:
            try:
                edges1.append(node_mapping[(br_parent[desc_z_id], desc_z_id-1)])
                edges2.append(first_halo_index)
            except:
                # If the parent branch does not exist, skip this edge
                bad_halo_idx.append(halo_indices)
                continue

        # Temporal edges within branch (consecutive node_ids)
        if num_halo_br >= 1:
            edges1.extend(halo_indices[:-1])
            edges2.extend(halo_indices[1:])
    edge_index = torch.tensor([edges1, edges2], dtype=torch.long)

    # Create PyTorch Geometric Data object
    pyg_data = Data(x=x, edge_index=edge_index)
    pyg_data = sort_tree(pyg_data, sort_prop=0) if sort else pyg_data

    return pyg_data

def find_main_branch(G, mass_dict):
    """
    Finds the most massive branch by following the highest-mass descendant at each step.

    Parameters:
    - G: NetworkX DiGraph representing the merger tree.
    - mass_dict: Dictionary mapping node IDs to their logarithmic mass.

    Returns:
    - List of nodes in the most massive branch.
    """
    root = min(G.nodes)  # Assuming root node has the smallest index
    main_branch = [root]

    while True:
        successors = list(G.successors(main_branch[-1]))
        if not successors:
            break
        # Select the most massive successor
        next_node = max(successors, key=lambda n: mass_dict[n])
        main_branch.append(next_node)

    return main_branch

def plot_merger_tree_pyg_color(data, ax=None, log_m_min=None, log_m_max=None):
    """
    Plots a merger tree from a PyTorch Geometric data object, coloring nodes based on log mass.

    Parameters:
    - data: PyTorch Geometric Data object with `data.x` (features) and `data.edge_index` (edges).
    - ax: Matplotlib axis (optional).
    """
    # Extract node features
    #  and edges
    log_m = data.x[:, 0].cpu().numpy()  # Ensure tensor is converted to NumPy
    edge_index = data.edge_index.cpu().numpy().T  # Convert to (parent, child) format

    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from(map(tuple, edge_index))

    # Use Graphviz layout (left-to-right positioning)
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    # Normalize node sizes based on log mass
    log_m_min = np.nanmin(log_m).item() if log_m_min is None else log_m_min
    log_m_max = np.nanmax(log_m).item() if log_m_max is None else log_m_max
    node_sizes = (log_m - log_m_min) / (log_m_max - log_m_min) * 50 + 4  # Avoid size=0 nodes

    # Normalize log mass for colormap
    norm = mcolors.Normalize(vmin=log_m_min, vmax=log_m_max)
    cmap = sns.color_palette("RdBu_r", as_cmap=True)
    node_colors = [cmap(norm(m)) for m in log_m]

    # Draw the graph
    if ax is None:
        fig, ax = plt.subplots()
    ax = nx.draw(
        G, pos, with_labels=False, node_size=node_sizes,
        node_color=node_colors, edge_color="black",
        font_size=12, arrows=False, ax=ax,
        edgecolors="black"
    )
