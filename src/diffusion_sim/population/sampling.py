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

    Supports Gaussian (normal) distribution with truncation to [0, 1].
    """

    @staticmethod
    def sample_normal(mean: float, std: float, low: float = 0.0, high: float = 1.0) -> float:
        """
        Sample from truncated normal distribution.

        Uses rejection sampling to ensure values are within [low, high].

        Args:
            mean: Mean of the distribution
            std: Standard deviation
            low: Lower bound (default 0.0)
            high: Upper bound (default 1.0)

        Returns:
            Sample in [low, high]
        """
        # Rejection sampling with max attempts
        max_attempts = 100
        for _ in range(max_attempts):
            value = random.gauss(mean, std)
            if low <= value <= high:
                return value

        # Fallback: clamp to bounds if rejection sampling fails
        return max(low, min(high, random.gauss(mean, std)))

    @staticmethod
    def sample_traits_from_params(trait_params: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Sample traits from Gaussian distributions defined by mean and std.

        Args:
            trait_params: Dict mapping trait name to {"mean": float, "std": float}

        Returns:
            Sampled traits dictionary
        """
        traits = {}
        for trait_name, params in trait_params.items():
            mean = params["mean"]
            std = params["std"]
            traits[trait_name] = TraitSampler.sample_normal(mean, std)
        return traits

    # Legacy methods for backward compatibility
    @staticmethod
    def sample_beta(alpha: float, beta: float) -> float:
        """Sample from Beta distribution."""
        return random.betavariate(alpha, beta)

    @staticmethod
    def sample_uniform(low: float, high: float) -> float:
        """Sample uniformly from [low, high]."""
        return random.uniform(low, high)

    @staticmethod
    def sample_traits_from_ranges(trait_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Sample traits from uniform ranges (legacy)."""
        traits = {}
        for trait_name, (low, high) in trait_ranges.items():
            traits[trait_name] = TraitSampler.sample_uniform(low, high)
        return traits


class PopulationGenerator:
    """
    Generator for heterogeneous agent populations.

    Supports stratified sampling based on Rogers' diffusion categories (4 categories):
    - Early Adopters (16%): High attitude/PBC (mean=0.8)
    - Early Majority (34%): Above average (mean=0.6)
    - Late Majority (34%): Below average (mean=0.4)
    - Laggards (16%): Low attitude/PBC (mean=0.2)

    Uses Gaussian distributions with mean and std for each trait.
    Conformity and risk_aversion are fixed to simplify the model.
    """

    # Rogers' distribution (4 categories)
    ROGERS_DISTRIBUTION = {
        "early_adopters": 0.16,   # Combines innovators (2.5%) + early adopters (13.5%)
        "early_majority": 0.34,
        "late_majority": 0.34,
        "laggards": 0.16,
    }

    # Default trait parameters: Gaussian (mean, std) for each category
    # Only 4 core traits; conformity and risk_aversion are fixed
    DEFAULT_TRAIT_PARAMS = {
        "early_adopters": {
            "attitude": {"mean": 0.80, "std": 0.10},
            "pbc": {"mean": 0.80, "std": 0.10},
            "share_propensity": {"mean": 0.85, "std": 0.10},
            "innovativeness": {"mean": 0.85, "std": 0.10},
        },
        "early_majority": {
            "attitude": {"mean": 0.60, "std": 0.10},
            "pbc": {"mean": 0.60, "std": 0.10},
            "share_propensity": {"mean": 0.65, "std": 0.10},
            "innovativeness": {"mean": 0.55, "std": 0.10},
        },
        "late_majority": {
            "attitude": {"mean": 0.40, "std": 0.10},
            "pbc": {"mean": 0.40, "std": 0.10},
            "share_propensity": {"mean": 0.45, "std": 0.10},
            "innovativeness": {"mean": 0.35, "std": 0.10},
        },
        "laggards": {
            "attitude": {"mean": 0.20, "std": 0.10},
            "pbc": {"mean": 0.20, "std": 0.10},
            "share_propensity": {"mean": 0.25, "std": 0.10},
            "innovativeness": {"mean": 0.15, "std": 0.10},
        },
    }

    # Fixed traits (not sampled)
    FIXED_TRAITS = {
        "conformity": 1.0,       # All agents equally sensitive to social norms
        "risk_aversion": 0.0,    # No risk aversion (since friction=0)
    }

    def __init__(
        self,
        distribution: Optional[Dict[str, float]] = None,
        trait_params: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize population generator.

        Args:
            distribution: Custom distribution (default: Rogers)
            trait_params: Custom trait parameters (mean, std) per category
            seed: Random seed
        """
        self.distribution = distribution or self.ROGERS_DISTRIBUTION
        self.trait_params = trait_params or self.DEFAULT_TRAIT_PARAMS

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
        Sample traits for a specific category using Gaussian distributions.

        Args:
            category: Category name

        Returns:
            Sampled traits dictionary (includes both sampled and fixed traits)
        """
        if category not in self.trait_params:
            raise ValueError(f"Unknown category: {category}")

        # Sample traits from Gaussian distributions
        params = self.trait_params[category]
        traits = TraitSampler.sample_traits_from_params(params)

        # Add fixed traits
        traits.update(self.FIXED_TRAITS)

        return traits

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
