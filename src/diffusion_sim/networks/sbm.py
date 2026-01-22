"""
Stochastic Block Model (SBM) network generator.

Used for modeling community-structured networks with clear block/cluster boundaries.
Suitable for brand sentiment diffusion where communities have distinct value orientations.
"""
import random
from typing import List, Dict, Optional
import networkx as nx
from ..core.schemas import NetworkConfig


class SBMNetworkBuilder:
    """
    Builder for Stochastic Block Model networks.

    Creates networks with community structure where:
    - Nodes within the same block have higher connection probability
    - Nodes across different blocks have lower connection probability
    - Useful for modeling value-based communities in sentiment diffusion
    """

    @staticmethod
    def build(
        n: int,
        num_blocks: int,
        p_within: float,
        p_between: float,
        block_sizes: Optional[List[int]] = None,
        seed: Optional[int] = None
    ) -> nx.Graph:
        """
        Build a Stochastic Block Model network.

        Args:
            n: Total number of nodes
            num_blocks: Number of blocks/communities
            p_within: Probability of edge within same block (high, e.g., 0.3)
            p_between: Probability of edge between different blocks (low, e.g., 0.01)
            block_sizes: List of block sizes (must sum to n), or None for equal sizes
            seed: Random seed for reproducibility

        Returns:
            NetworkX Graph with community structure
        """
        if seed is not None:
            random.seed(seed)

        # Determine block assignments
        if block_sizes is None:
            # Equal-sized blocks
            base_size = n // num_blocks
            remainder = n % num_blocks
            block_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_blocks)]
        else:
            if sum(block_sizes) != n:
                raise ValueError(f"Block sizes must sum to n={n}, got {sum(block_sizes)}")
            if len(block_sizes) != num_blocks:
                raise ValueError(f"Number of block sizes must equal num_blocks={num_blocks}")

        # Assign nodes to blocks
        node_blocks = []
        for block_id, size in enumerate(block_sizes):
            node_blocks.extend([block_id] * size)

        # Create graph
        graph = nx.Graph()
        graph.add_nodes_from(range(n))

        # Add block assignments as node attributes
        for node_id, block_id in enumerate(node_blocks):
            graph.nodes[node_id]['block'] = block_id

        # Add edges based on SBM probabilities
        for i in range(n):
            for j in range(i + 1, n):
                # Determine edge probability
                if node_blocks[i] == node_blocks[j]:
                    p_edge = p_within
                else:
                    p_edge = p_between

                # Add edge with probability p_edge
                if random.random() < p_edge:
                    graph.add_edge(i, j)

        return graph

    @staticmethod
    def build_from_config(config: Dict) -> nx.Graph:
        """
        Build SBM network from configuration dictionary.

        Args:
            config: Configuration with keys:
                - n: number of nodes
                - num_blocks: number of communities
                - p_within: within-block edge probability
                - p_between: between-block edge probability
                - block_sizes: optional list of block sizes
                - seed: optional random seed

        Returns:
            NetworkX Graph
        """
        return SBMNetworkBuilder.build(
            n=config['n'],
            num_blocks=config['num_blocks'],
            p_within=config['p_within'],
            p_between=config['p_between'],
            block_sizes=config.get('block_sizes'),
            seed=config.get('seed')
        )

    @staticmethod
    def get_network_stats(graph: nx.Graph) -> Dict:
        """
        Compute statistics for SBM network.

        Args:
            graph: SBM network graph

        Returns:
            Dictionary with network statistics
        """
        # Basic stats
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        avg_degree = 2 * m / n if n > 0 else 0

        # Block statistics
        blocks = {}
        for node in graph.nodes():
            block_id = graph.nodes[node].get('block', 0)
            if block_id not in blocks:
                blocks[block_id] = []
            blocks[block_id].append(node)

        num_blocks = len(blocks)
        block_sizes = {bid: len(nodes) for bid, nodes in blocks.items()}

        # Modularity (community structure quality)
        try:
            communities = [set(nodes) for nodes in blocks.values()]
            modularity = nx.community.modularity(graph, communities)
        except:
            modularity = None

        # Clustering coefficient
        try:
            clustering = nx.average_clustering(graph)
        except:
            clustering = 0

        # Average shortest path (if connected)
        try:
            if nx.is_connected(graph):
                avg_path = nx.average_shortest_path_length(graph)
            else:
                # For disconnected graphs, use largest component
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                avg_path = nx.average_shortest_path_length(subgraph)
        except:
            avg_path = None

        return {
            'num_nodes': n,
            'num_edges': m,
            'avg_degree': avg_degree,
            'num_blocks': num_blocks,
            'block_sizes': block_sizes,
            'modularity': modularity,
            'clustering_coefficient': clustering,
            'avg_shortest_path': avg_path,
            'connected': nx.is_connected(graph),
        }
