import numpy as np
import scipy as sp
import networkx as nx
import random
from collections import Counter
from scipy.sparse.linalg import eigs
import itertools
import random
import heapq 


def get_B0(n,node_degree,lap):
    offset = 2 * np.max(node_degree)
    T = offset * sp.sparse.eye(n, format="csc") - lap
    lk, Uk = np.linalg.eigh(T.toarray(),UPLO='L')
    # some problems with this function
    # lk, Uk = sp.sparse.linalg.eigsh(
    #     T, k=preserved_eig_size, which="LM", tol=1e-5,ncv=2*preserved_eig_size
    # )
    lk = (offset - lk)[::-1]
    Uk = Uk[:, ::-1]

    # compute L^-1/2
    mask = lk < 1e-5
    lk[mask] = 1
    lk_inv = 1 / np.sqrt(lk)
    lk_inv[mask] = 0
    return sp.sparse.csr_matrix(Uk * lk_inv[np.newaxis, :])  # = Uk @ np.diag(lk_inv)

def get_edge_contraction_sets(adj):
    us, vs, _ = sp.sparse.find(sp.sparse.triu(adj))
    return np.stack([us, vs], axis=1)

def get_edge_local_variation_cost(edge, args=None):
    """Compute the local variation cost for an edge"""
    adj, node_degree, A = args
    u, v = edge
    w = adj[u, v]
    
    L = sp.sparse.csr_matrix([[2 * node_degree[u] - w, -w], [-w, 2 * node_degree[v] - w]])
    B = A[edge, :]
    return sp.sparse.linalg.norm(B.T @ L @ B)  

def get_node_contraction_sets(adj) -> list:
    """Returns neighborhood contraction sets"""
    adj_with_self_loops = adj.copy().tolil()
    adj_with_self_loops.setdiag(1)
    return [np.array(nbrs) for nbrs in adj_with_self_loops.rows]
 
def get_node_local_variation_cost(nodes, args=None):
    """Compute the local variation cost for a set of nodes"""
    adj, node_degree, A = args
    nc = len(nodes)
    if nc == 1:
        return float('inf')

    ones = np.ones(nc)
    W = adj[nodes, :][:, nodes]
    L = np.diag(2 * node_degree[nodes] - W @ ones) - W
    B = (np.eye(nc) - np.outer(ones, ones) / nc) @ A[nodes, :]
    return np.linalg.norm(B.T @ L @ B) / (nc - 1)

def max_count_func(label):
    count = Counter(label)
    return count.most_common(1)[0][0]

def get_lap_coresed_graph(G, reduce_frac=.5, disturb_frac=.0, max_condense_size=5, reindex=False):    
    if disturb_frac>0:
        reduce_frac = reduce_frac + (random.uniform(-1, 1) * disturb_frac) 

    adj = nx.adjacency_matrix(G, dtype=np.float32)
    node_degree = adj.sum(0)
    lap = sp.sparse.diags(node_degree) - adj
    n = adj.shape[0]
    B0 = get_B0(n, node_degree, lap)

    contraction_sets = get_edge_contraction_sets(adj)
    costs = np.apply_along_axis(get_edge_local_variation_cost, 1, contraction_sets, (adj, node_degree, B0))
    perm = costs.argsort()
    contraction_sets = contraction_sets[perm]

    node_mask = np.zeros(n, dtype=bool)
    node_map = [[i] for i in range(n)]

    num_reduce = int(n * reduce_frac)
    i = 0
    contraction_count = 0
    while contraction_count < num_reduce and i < len(contraction_sets):
        condensed_node, corased_node = contraction_sets[i]
        if condensed_node == corased_node or len(node_map[condensed_node]) + len(node_map[corased_node]) > max_condense_size:
            i += 1
            continue

        if not node_mask[condensed_node]:
            node_map[condensed_node].extend(node_map[corased_node])
            node_mask[corased_node] = True
            contraction_sets[contraction_sets == corased_node] = condensed_node
            contraction_count += 1
        i += 1

    new_nodes = np.arange(n, dtype=int)[~node_mask]
    new_edges, edge_counts = np.unique(contraction_sets, axis=0, return_counts=True)

    mask = new_edges[:, 0] != new_edges[:, 1]
    no_loop_edges = new_edges[mask]



    reduced_G = nx.Graph()
    reduced_G.add_nodes_from(new_nodes)
    node_map = [node_map[i] for i in range(n) if not node_mask[i]]
    new_node_counts = np.array([len(nodes) for nodes in node_map])
    nx.set_node_attributes(reduced_G, dict(zip(new_nodes, new_node_counts)), 'node_count')
    nx.set_node_attributes(reduced_G, dict(zip(new_nodes, node_map)), 'node_map')
    nx.set_node_attributes(reduced_G, 0, 'ring_num')

    reduced_G.add_edges_from(no_loop_edges)
    self_loops = new_edges[~mask]
    loops_count = edge_counts[~mask]
    for i, edge in enumerate(self_loops):
        node = edge[0]
        reduced_G.nodes[node]['ring_num'] = loops_count[i] - (reduced_G.nodes[node]['node_count']-1)
    if reindex:
        reduced_G = nx.convert_node_labels_to_integers(reduced_G)
    if len(G.graph) > 0:
        for (key,val) in  G.graph.items():
            reduced_G.graph[key] = val
    return reduced_G

def find_ring_clusters(G, split_ring):
    """使用 NetworkX 将共享结点的环分为同一簇"""
    # cycles = list(nx.cycle_basis(G))
    cycles = list(set(c) for c in nx.cycle_basis(G) if len(c) <= 6)  # 获取所有环
    
    # 将每个环作为节点，若两个环共享节点则连接它们
    ring_graph = nx.Graph()
    for i, cycle in enumerate(cycles):
        ring_graph.add_node(i, node_map=set(cycle))
    for i in range(len(cycles)):
        for j in range(i + 1, len(cycles)):
            c1, c2 = cycles[i], cycles[j]
            if (c1 & c2): ring_graph.add_edge(i, j)

    if split_ring:
        for component in list(nx.connected_components(ring_graph)):
            if len(component) > 2:
                component = np.array(list(component))

                # star eliminate
                centered_rings = np.array(list(ring_graph.degree(component)))[:,1] > 2
                ring_graph.remove_nodes_from(component[centered_rings])
                component = component[~centered_rings]

                # chain eliminate
                for ring_index in component:
                    anchor_neighbors = list(ring_graph.neighbors(ring_index))
                    if (len(anchor_neighbors) > 1):
                        ring_graph.remove_node(ring_index)

    ring_clusters = []
    ring_counts = []
    for component in nx.connected_components(ring_graph):
        cluster_atoms = set()        
        for i,ring_index in enumerate(component):
            cluster_atoms.update(ring_graph.nodes[ring_index]['node_map'])
        ring_clusters.append(np.array(list(cluster_atoms),dtype=np.int32))
        ring_counts.append(len(component))
    
    return ring_clusters, ring_counts

def get_expand_check(G, coarsed_G, node_valency):
    bond_valency = np.array([0, 1, 2, 3, 1.5])
    node_valency = np.array(node_valency)
    is_ring = np.array(list(nx.get_node_attributes(coarsed_G,'ring_num').values()),dtype=bool)
    clusters = list(nx.get_node_attributes(coarsed_G,'node_map').values())

    nx.set_node_attributes(coarsed_G, True, 'allow_extend')

    for i, cluster in enumerate(clusters):
        if is_ring[i]: continue
        sub_G = nx.subgraph(G, cluster)
        atoms = list(nx.get_node_attributes(sub_G, 'label').values())
        allowed_valency = node_valency[atoms].sum()
        if len(cluster) > 1: 
            internal_bonds = list(nx.get_edge_attributes(sub_G, 'bond_type').values())
            internal_valencies = bond_valency[internal_bonds].sum() * 2
            allowed_valency = allowed_valency - internal_valencies
        outside_valencies = 0
        for node in cluster:
            neighbors = set(G.neighbors(node)) - set(cluster)  # 外部邻居节点
            for neighbor in neighbors:
                bond_type = G.get_edge_data(node, neighbor).get('bond_type', 0)
                outside_valencies += bond_valency[bond_type]
        allowed_valency -= outside_valencies
        assert allowed_valency >= 0, print('Overall valency has exceed the valency allowed, check the valency list')
        if allowed_valency == 0: coarsed_G.nodes[cluster[0]]['allow_extend'] = False
    return coarsed_G

def get_custom_coarsed_graph(G, split_ring=False, reindex=False):
    adj = nx.adjacency_matrix(G, dtype=np.float32)
    node_degree = adj.sum(0)
    lap = sp.sparse.diags(node_degree) - adj
    n = adj.shape[0]
    B0 = get_B0(n, node_degree, lap)

    ring_partitions, ring_counts = find_ring_clusters(G, split_ring)
    
    G_without_rings = G.copy()
    if len(ring_partitions) > 0:
        ring_nodes = np.concatenate(ring_partitions,-1)
        G_without_rings.remove_nodes_from(ring_nodes)
        G_without_rings.add_nodes_from(ring_nodes)

    partitions = []
    if G_without_rings.number_of_edges() > 0:
        contraction_adj = nx.adjacency_matrix(G_without_rings, nodelist=G.nodes, dtype=np.float32)
        contraction_sets = get_node_contraction_sets(contraction_adj)

        # 初始化最小堆
        cost_heap = []
        for contraction_set in contraction_sets:
            cost = get_node_local_variation_cost(contraction_set, (contraction_adj, node_degree, B0))
            heapq.heappush(cost_heap, (cost, set(contraction_set)))

        marked = np.zeros(n, dtype=bool)
        while len(cost_heap):  # 当堆不为空时继续
            cost, contraction_set = heapq.heappop(cost_heap)  # 弹出成本最低的集合
            if len(contraction_set) > 1:
                contraction_set = np.array(list(contraction_set))
                marks = marked[contraction_set]
                if not np.any(marks):
                    partitions.append(contraction_set)
                    marked[contraction_set] = True
                else:
                    contraction_set = contraction_set[~marks]
                    if len(contraction_set) > 1:
                        cost = get_node_local_variation_cost(contraction_set, (contraction_adj, node_degree, B0))
                        heapq.heappush(cost_heap, (cost, set(contraction_set)))  # 重新插入成本
    
    P = np.eye(n)
    mask = np.ones(n, dtype=bool)
    partitions = ring_partitions + partitions
    node_map = {}
    node_count = {}
    ring_nums = {}
    for i, partition in enumerate(partitions):
        coarsed_node = partition[0]
        P[coarsed_node, partition] = 1
        mask[partition[1:]] = False
        node_map[coarsed_node] = partition
        node_count[coarsed_node] = len(partition)
    P = P[mask, :]
    for i, partition in enumerate(ring_partitions):
        ring_nums[partition[0]] = ring_counts[i]
    node_assign = {i:assign for i,assign in enumerate(np.where(mask)[0])}
    
    coarsed_adj = (P @ adj @ P.T).astype(bool)
    coarsed_adj = coarsed_adj * ~np.eye(coarsed_adj.shape[0],dtype=bool) # remove self loops

    if coarsed_adj.size == 1:
        coarsed_G = nx.Graph()
        coarsed_G.add_node(0)
    else:
        coarsed_G = nx.from_numpy_array(coarsed_adj)

    # initialize attrs
    coarsed_G = nx.relabel_nodes(coarsed_G, node_assign)
    nx.set_node_attributes(coarsed_G, 1, 'node_count')
    nx.set_node_attributes(coarsed_G, 0, 'ring_num')
    nx.set_node_attributes(coarsed_G, {n:[n] for n in coarsed_G.nodes}, 'node_map')

    # assign attrs
    nx.set_node_attributes(coarsed_G, node_count, 'node_count')
    nx.set_node_attributes(coarsed_G, node_map, 'node_map')
    nx.set_node_attributes(coarsed_G, ring_nums, 'ring_num')

    if 'y' in G.graph:
        coarsed_G.graph['y'] = G.graph['y']
    if reindex:
        coarsed_G = nx.convert_node_labels_to_integers(coarsed_G)
    return coarsed_G

def get_comm_coarsed_graph(G, 
                           reduce_frac=1.,
                           max_condense_size=float('inf'), 
                           min_condense_size=2,
                           resolution=1, 
                           reindex=False, 
                           ):   
    communities = nx.algorithms.community.louvain_communities(G, resolution=resolution, seed=0)
    
    communities = sorted(communities, key=len, reverse=True)

    # else: raise NotImplementedError
    coarsed_G = G.copy()
    node_attributes = {node: {'node_map': [node], 'node_count': 1, 'ring_num': 0} for node in coarsed_G.nodes}
    nx.set_node_attributes(coarsed_G, node_attributes)

    i = 0
    graph_size = G.number_of_nodes()
    while i < len(communities) and coarsed_G.number_of_nodes() > graph_size * (1-reduce_frac):
        comm = list(communities[i])
        coarsed_node_idx = comm[0]
        comm_size = len(comm)
        i += 1
        if comm_size > max_condense_size or comm_size < min_condense_size: continue

        comm_neighbors = set()
        for node in comm:
            neighbors = coarsed_G.neighbors(node)
            comm_neighbors.update(n for n in neighbors if n not in comm)

        coarsed_G.remove_nodes_from(comm)

        ring_num = G.subgraph(comm).number_of_edges() - (comm_size - 1)
        if ring_num < 0:
            print("Encounter community with negtive ring count, variables info:")
            print("Community:", comm)
            print("Edge in community:", G.subgraph(comm).number_of_edges())
            print("Community size:", comm_size)
            continue
        
        coarsed_G.add_node(coarsed_node_idx, 
                           node_map=list(comm), 
                           node_count=comm_size,
                           ring_num= ring_num
                           )
        
        coarsed_G.add_edges_from((neighbor, coarsed_node_idx) for neighbor in comm_neighbors)
    if reindex:
        coarsed_G = nx.convert_node_labels_to_integers(coarsed_G)
    return coarsed_G