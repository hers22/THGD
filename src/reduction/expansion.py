import numpy as np
import scipy as sp
import networkx as nx
from scipy.sparse import coo_array, csr_array, eye
import random
from collections import Counter
import torch
from src.reduction.structural_types import NORMAL, CONNECTION, CONDENSED


def connect_neighbors(neighbors, node_map):
    connected_Nei = nx.algorithms.bipartite.complete_bipartite_graph(neighbors, node_map)
    nx.set_edge_attributes(connected_Nei,CONNECTION,'type_mask')
    nx.set_edge_attributes(connected_Nei,0,'edge_count')
    return connected_Nei.edges(data=True)

def add_sub_graph(node_map, ring_num):
    comp_G = nx.complete_graph(node_map)
    nx.set_node_attributes(comp_G,CONDENSED + ring_num,'type_mask') 
    nx.set_edge_attributes(comp_G,CONDENSED + ring_num,'type_mask')
    nx.set_edge_attributes(comp_G, 0 ,'edge_count')
    return comp_G
    
def get_expanded_graph(reduced_G,reindex=False):
    '''
    expand the reduced graph with randomly generated tree
    reduced_G: nx.Graph() with 'value' attr
    expand_type: 'tree', 'complete'
    '''
    nodes = list(reduced_G.nodes)
    n = len(nodes)
    node_counts = [int(item[1]) for item in reduced_G.nodes(data='node_count')]
    node_maps = [item[1] for item in reduced_G.nodes(data='node_map')]
    ring_nums = [item[1] for item in reduced_G.nodes(data='ring_num')]
    enable_map = node_maps[0] is not None
    
    
    expanded_G = reduced_G.copy()
    nx.set_node_attributes(expanded_G,NORMAL,'type_mask') 
    nx.set_edge_attributes(expanded_G,NORMAL,'type_mask') 

    offset = n + 1
    for i, node_count in enumerate(node_counts):
        if node_count > 1:
            node_to_expand = nodes[i]
            node_map = node_maps[i] if enable_map else list(range(offset, offset + node_count))
            offset += node_count

            sub_G = add_sub_graph(node_map, ring_nums[i])
            src_nodes = np.array(list(expanded_G.neighbors(node_to_expand)))   # find the neighbor of the condensed node
            expanded_G.remove_node(node_to_expand) # remove the condensed node

            # fully connect neighbors
            neighbor_edges = connect_neighbors(src_nodes, node_map)

            expanded_G.add_nodes_from(sub_G.nodes(data=True))
            expanded_G.add_edges_from(sub_G.edges(data=True))
            expanded_G.add_edges_from(neighbor_edges)
    
    # sort the node values of expanded G
    sorted_expanded_G = nx.Graph()
    sorted_expanded_G.add_nodes_from(sorted(expanded_G.nodes(data=True)))
    sorted_expanded_G.add_edges_from(expanded_G.edges(data=True))
    if reindex:
        sorted_expanded_G = nx.convert_node_labels_to_integers(sorted_expanded_G)
    if len(reduced_G.graph) > 0:
        for (key,val) in reduced_G.graph.items():
            sorted_expanded_G.graph[key] = val
    return sorted_expanded_G