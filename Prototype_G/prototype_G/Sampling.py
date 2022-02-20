import numpy as np


def sampling(src_nodes, sample_num, neighbor_table):
    """Sampling a specified number of neighbouring nodes based on the source node, noting that the sampling is done with a putback;
    if the number of neighbours of a node is less than the number of nodes sampled, duplicate nodes will be sampled
    
    Arguments:
        src_nodes {list, ndarray} -- List of source nodes
        sample_num {int} -- Number of nodes to be sampled
        neighbor_table {dict} -- Mapping table of nodes to their neighbours
    
    Returns:
        np.ndarray -- List of sampling results composition
    """
    results = []
    for sid in src_nodes:
        # Sampling with playback from the node's neighbours

        res = np.random.choice(neighbor_table, size=(sample_num, ))
        results.append(res)

    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """Multi-order sampling based on source nodes
    
    Arguments:
        src_nodes {list, np.ndarray} -- Source node id
        sample_nums {list of int} -- Number of nodes to be sampled
        neighbor_table {dict} -- Mapping of a node to its neighbouring nodes
    
    Returns:
        [list of ndarray] -- Results per order of sampling
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        neighbor=[]
        for i in range(600):
            if neighbor_table[k][i+1]==1:
                neighbor.append(i)
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor)
        sampling_result.append(hopk_result)
    return sampling_result
