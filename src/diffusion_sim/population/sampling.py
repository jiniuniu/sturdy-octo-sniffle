"""
Trait sampling for agent population generation.

Samples traits from statistical distributions to create heterogeneous agents.
Follows Rogers' diffusion of innovations categories.
"""
import random
from typing import Dict, List, Optional, Tuple
import networkx as nx
from ..agents.tpb_agent import TPBAgent


class TraitSampler:
    """
    Sampler for agent traits from prior distributions.

    Uses Beta distributions for bounded [0, 1] traits.
    """

    @staticmethod
    def sample_beta(alpha: float, beta: float) -> float:
        """
        Sample from Beta distribution.

        Args:
            alpha: Alpha parameter (shape)
            beta: Beta parameter (shape)

        Returns:
            Sample in [0, 1]
        """
        return random.betavariate(alpha, beta)

    @staticmethod
    def sample_uniform(low: float, high: float) -> float:
        """
        Sample uniformly from [low, high].

        Args:
            low: Lower bound
            high: Upper bound

        Returns:
            Sample in [low, high]
        """
        return random.uniform(low, high)

    @staticmethod
    def sample_traits_from_ranges(trait_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Sample traits from uniform ranges.

        Args:
            trait_ranges: Dict mapping trait name to (min, max) tuple

        Returns:
            Sampled traits dictionary
        """
        traits = {}
        for trait_name, (low, high) in trait_ranges.items():
            traits[trait_name] = TraitSampler.sample_uniform(low, high)
        return traits

    @staticmethod
    def sample_traits_from_beta_params(
        trait_params: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Sample traits from Beta distributions.

        Args:
            trait_params: Dict mapping trait name to (alpha, beta) tuple

        Returns:
            Sampled traits dictionary
        """
        traits = {}
        for trait_name, (alpha, beta) in trait_params.items():
            traits[trait_name] = TraitSampler.sample_beta(alpha, beta)
        return traits


class PopulationGenerator:
    """
    Generator for heterogeneous agent populations.

    Supports stratified sampling based on Rogers' diffusion categories (4 categories):
    - Early Adopters (16%): High innovativeness, low risk aversion (combines innovators + early adopters)
    - Early Majority (34%): Slightly above average, moderate conformity
    - Late Majority (34%): Below average, high conformity
    - Laggards (16%): Low innovativeness, high risk aversion
    """

    # Rogers' distribution (4 categories)
    ROGERS_DISTRIBUTION = {
        "early_adopters": 0.16,   # Combines innovators (2.5%) + early adopters (13.5%)
        "early_majority": 0.34,
        "late_majority": 0.34,
        "laggards": 0.16,
    }

    # Default trait ranges for each category
    DEFAULT_TRAIT_RANGES = {
        "early_adopters": {
            # Merged range from innovators and early adopters
            "attitude": (0.6, 0.95),
            "pbc": (0.6, 0.9),
            "conformity": (0.1, 0.6),
            "risk_aversion": (0.0, 0.5),
            "share_propensity": (0.6, 1.0),
            "innovativeness": (0.65, 1.0),
        },
        "early_majority": {
            "attitude": (0.5, 0.7),
            "pbc": (0.5, 0.7),
            "conformity": (0.5, 0.8),
            "risk_aversion": (0.3, 0.6),
            "share_propensity": (0.5, 0.8),
            "innovativeness": (0.45, 0.65),
        },
        "late_majority": {
            "attitude": (0.3, 0.6),
            "pbc": (0.3, 0.6),
            "conformity": (0.6, 0.9),
            "risk_aversion": (0.5, 0.8),
            "share_propensity": (0.3, 0.6),
            "innovativeness": (0.25, 0.5),
        },
        "laggards": {
            "attitude": (0.1, 0.4),
            "pbc": (0.2, 0.5),
            "conformity": (0.3, 0.7),
            "risk_aversion": (0.7, 1.0),
            "share_propensity": (0.1, 0.5),
            "innovativeness": (0.0, 0.3),
        },
    }

    def __init__(
        self,
        distribution: Optional[Dict[str, float]] = None,
        trait_ranges: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize population generator.

        Args:
            distribution: Custom distribution (default: Rogers)
            trait_ranges: Custom trait ranges per category
            seed: Random seed
        """
        self.distribution = distribution or self.ROGERS_DISTRIBUTION
        self.trait_ranges = trait_ranges or self.DEFAULT_TRAIT_RANGES

        if seed is not None:
            random.seed(seed)

        # Validate distribution sums to 1
        total = sum(self.distribution.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Distribution must sum to 1, got {total}")

    def generate_population(self, graph: nx.Graph) -> None:
        """
        Generate heterogeneous population and assign traits to all nodes.

        Args:
            graph: Network graph with nodes initialized
        """
        n_nodes = graph.number_of_nodes()
        node_ids = list(graph.nodes())

        # Shuffle for randomness
        random.shuffle(node_ids)

        # Assign nodes to categories
        idx = 0
        for category, ratio in self.distribution.items():
            n_category = int(n_nodes * ratio)

            # Handle last category to ensure all nodes assigned
            if category == list(self.distribution.keys())[-1]:
                n_category = n_nodes - idx

            category_nodes = node_ids[idx:idx + n_category]

            # Sample and assign traits
            for node_id in category_nodes:
                traits = self._sample_traits_for_category(category)
                TPBAgent.initialize_agent(graph, node_id, traits)

            idx += n_category

    def _sample_traits_for_category(self, category: str) -> Dict[str, float]:
        """
        Sample traits for a specific category.

        Args:
            category: Category name

        Returns:
            Sampled traits dictionary
        """
        if category not in self.trait_ranges:
            raise ValueError(f"Unknown category: {category}")

        ranges = self.trait_ranges[category]
        return TraitSampler.sample_traits_from_ranges(ranges)

    def get_category_distribution(self, graph: nx.Graph) -> Dict[str, int]:
        """
        Get actual distribution of categories in population.

        Args:
            graph: Network graph with agents

        Returns:
            Dictionary of category counts
        """
        counts = {cat: 0 for cat in self.distribution.keys()}

        for node_id in graph.nodes():
            traits = TPBAgent.get_all_traits(graph, node_id)
            category = TPBAgent.categorize_agent(traits)
            if category in counts:
                counts[category] += 1

        return counts

    def select_initial_adopters(
        self,
        graph: nx.Graph,
        n_initial: int,
        strategy: str = "innovators"
    ) -> List[int]:
        """
        Select initial adopters based on strategy.

        Args:
            graph: Network graph
            n_initial: Number of initial adopters
            strategy: Selection strategy
                - "innovators" or "early_adopters": Select agents with highest innovativeness
                - "random": Random selection
                - "high_degree": Select high-degree nodes (hubs)

        Returns:
            List of selected node IDs
        """
        node_ids = list(graph.nodes())

        if strategy in ["innovators", "early_adopters"]:
            # Select agents with highest innovativeness (early adopters in 4-category system)
            candidates = []
            for node_id in node_ids:
                traits = TPBAgent.get_all_traits(graph, node_id)
                innovativeness = traits.get("innovativeness", 0.5)
                candidates.append((node_id, innovativeness))

            # Sort by innovativeness and take top N
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [node_id for node_id, _ in candidates[:n_initial]]

        elif strategy == "random":
            return random.sample(node_ids, n_initial)

        elif strategy == "high_degree":
            # Select nodes with highest degree
            degrees = dict(graph.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            return [node_id for node_id, _ in sorted_nodes[:n_initial]]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")
