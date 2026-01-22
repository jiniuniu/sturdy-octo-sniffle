"""
Sentiment trait sampling for brand crisis diffusion population generation.

Samples traits from statistical distributions to create heterogeneous agents
with different sentiment formation and expression behaviors.

Population Types (4 categories):
- Strong Critics (20%): High moral sensitivity, low expression threshold
- Conformists (40%): High conformity sensitivity, moderate expression threshold
- Silent Observers (25%): High expression threshold (unlikely to express)
- Amplifiers (15%): High share propensity, low social risk aversion
"""
import random
from typing import Dict, List, Optional
import networkx as nx
from ..agents.sentiment_agent import SentimentAgent
from .sampling import TraitSampler


class SentimentPopulationGenerator:
    """
    Generator for heterogeneous sentiment agent populations.

    Uses Gaussian distributions with (mean, std) for each trait.
    """

    # Population distribution (4 types)
    POPULATION_DISTRIBUTION = {
        "strong_critics": 0.20,      # Vocal critics with strong moral stance
        "conformists": 0.40,          # Follow neighborhood sentiment
        "silent_observers": 0.25,     # Aware but unlikely to express
        "amplifiers": 0.15,           # High engagement, spread sentiment
    }

    # Default trait parameters: Gaussian (mean, std) for each category
    # 7 traits per agent:
    # - attitude_baseline: Base attitude toward brand (0=negative, 1=positive)
    # - moral_sensitivity: Sensitivity to moral/ethical issues
    # - conformity_sensitivity: Sensitivity to social influence
    # - expression_threshold: Barrier to public expression
    # - social_risk_aversion: Aversion to expressing controversial views
    # - negativity_bias: Tendency to focus on negative information
    # - share_propensity: Willingness to share content
    DEFAULT_TRAIT_PARAMS = {
        "strong_critics": {
            "attitude_baseline": {"mean": 0.30, "std": 0.10},      # Skeptical of brand
            "moral_sensitivity": {"mean": 0.85, "std": 0.10},      # High moral standards
            "conformity_sensitivity": {"mean": 0.30, "std": 0.10}, # Independent thinkers
            "expression_threshold": {"mean": 0.25, "std": 0.10},   # Low barrier to express
            "social_risk_aversion": {"mean": 0.20, "std": 0.10},   # Willing to take risks
            "negativity_bias": {"mean": 0.80, "std": 0.10},        # Focus on negatives
            "share_propensity": {"mean": 0.75, "std": 0.10},       # Active sharers
        },
        "conformists": {
            "attitude_baseline": {"mean": 0.50, "std": 0.15},      # Neutral baseline
            "moral_sensitivity": {"mean": 0.50, "std": 0.10},      # Moderate sensitivity
            "conformity_sensitivity": {"mean": 0.85, "std": 0.10}, # Highly conformist
            "expression_threshold": {"mean": 0.55, "std": 0.10},   # Moderate barrier
            "social_risk_aversion": {"mean": 0.60, "std": 0.10},   # Risk-averse
            "negativity_bias": {"mean": 0.50, "std": 0.10},        # Balanced
            "share_propensity": {"mean": 0.55, "std": 0.10},       # Moderate sharing
        },
        "silent_observers": {
            "attitude_baseline": {"mean": 0.45, "std": 0.15},      # Slightly skeptical
            "moral_sensitivity": {"mean": 0.60, "std": 0.10},      # Aware of issues
            "conformity_sensitivity": {"mean": 0.50, "std": 0.10}, # Moderate conformity
            "expression_threshold": {"mean": 0.80, "std": 0.10},   # High barrier
            "social_risk_aversion": {"mean": 0.85, "std": 0.10},   # Very risk-averse
            "negativity_bias": {"mean": 0.55, "std": 0.10},        # Slightly negative
            "share_propensity": {"mean": 0.30, "std": 0.10},       # Low sharing
        },
        "amplifiers": {
            "attitude_baseline": {"mean": 0.50, "std": 0.20},      # Varied baseline
            "moral_sensitivity": {"mean": 0.65, "std": 0.10},      # Moderately sensitive
            "conformity_sensitivity": {"mean": 0.60, "std": 0.10}, # Somewhat conformist
            "expression_threshold": {"mean": 0.40, "std": 0.10},   # Low-moderate barrier
            "social_risk_aversion": {"mean": 0.25, "std": 0.10},   # Low risk aversion
            "negativity_bias": {"mean": 0.60, "std": 0.10},        # Moderately negative
            "share_propensity": {"mean": 0.90, "std": 0.05},       # Very high sharing
        },
    }

    def __init__(
        self,
        distribution: Optional[Dict[str, float]] = None,
        trait_params: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize sentiment population generator.

        Args:
            distribution: Custom distribution (default: POPULATION_DISTRIBUTION)
            trait_params: Custom trait parameters (mean, std) per category
            seed: Random seed
        """
        self.distribution = distribution or self.POPULATION_DISTRIBUTION
        self.trait_params = trait_params or self.DEFAULT_TRAIT_PARAMS

        if seed is not None:
            random.seed(seed)

        # Validate distribution sums to 1
        total = sum(self.distribution.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Distribution must sum to 1, got {total}")

    def generate_population(self, graph: nx.Graph) -> None:
        """
        Generate heterogeneous sentiment population and assign traits to all nodes.

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

            # Sample and assign traits with category
            for node_id in category_nodes:
                traits = self._sample_traits_for_category(category)
                SentimentAgent.initialize_agent(graph, node_id, traits, category=category)

            idx += n_category

    def _sample_traits_for_category(self, category: str) -> Dict[str, float]:
        """
        Sample traits for a specific category using Gaussian distributions.

        Args:
            category: Category name

        Returns:
            Sampled traits dictionary
        """
        if category not in self.trait_params:
            raise ValueError(f"Unknown category: {category}")

        # Sample traits from Gaussian distributions
        params = self.trait_params[category]
        traits = TraitSampler.sample_traits_from_params(params)

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
            # Use category assigned during sampling
            category = SentimentAgent.get_category(graph, node_id)

            # Fallback to trait-based categorization for legacy agents
            if category is None:
                traits = SentimentAgent.get_all_traits(graph, node_id)
                category = SentimentAgent.categorize_agent(traits)

            if category in counts:
                counts[category] += 1

        return counts

    def select_initial_aware(
        self,
        graph: nx.Graph,
        n_initial: int,
        strategy: str = "random"
    ) -> List[int]:
        """
        Select initial aware agents (seed set for crisis awareness).

        Args:
            graph: Network graph
            n_initial: Number of initial aware agents
            strategy: Selection strategy
                - "random": Random selection
                - "high_degree": Select high-degree nodes (influencers)
                - "critics": Select strong critics
                - "amplifiers": Select amplifiers

        Returns:
            List of selected node IDs
        """
        node_ids = list(graph.nodes())

        if strategy == "random":
            return random.sample(node_ids, n_initial)

        elif strategy == "high_degree":
            # Select nodes with highest degree
            degrees = dict(graph.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            return [node_id for node_id, _ in sorted_nodes[:n_initial]]

        elif strategy == "critics":
            # Select strong critics (high moral sensitivity, low expression threshold)
            candidates = []
            for node_id in node_ids:
                traits = SentimentAgent.get_all_traits(graph, node_id)
                moral_sensitivity = traits.get("moral_sensitivity", 0.5)
                expression_threshold = traits.get("expression_threshold", 0.5)
                # Prioritize high moral sensitivity and low expression threshold
                score = moral_sensitivity * (1.0 - expression_threshold)
                candidates.append((node_id, score))

            # Sort by score and take top N
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [node_id for node_id, _ in candidates[:n_initial]]

        elif strategy == "amplifiers":
            # Select amplifiers (high share propensity, low social risk aversion)
            candidates = []
            for node_id in node_ids:
                traits = SentimentAgent.get_all_traits(graph, node_id)
                share_propensity = traits.get("share_propensity", 0.5)
                social_risk_aversion = traits.get("social_risk_aversion", 0.5)
                # Prioritize high share propensity and low risk aversion
                score = share_propensity * (1.0 - social_risk_aversion)
                candidates.append((node_id, score))

            # Sort by score and take top N
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [node_id for node_id, _ in candidates[:n_initial]]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def get_polarity_statistics(self, graph: nx.Graph) -> Dict[str, Dict[str, float]]:
        """
        Get polarity statistics by category.

        Args:
            graph: Network graph with sentiment agents

        Returns:
            Dictionary with polarity stats per category
        """
        category_polarities = {cat: [] for cat in self.distribution.keys()}

        for node_id in graph.nodes():
            category = SentimentAgent.get_category(graph, node_id)
            polarity = SentimentAgent.get_polarity(graph, node_id)

            if category and polarity is not None:
                category_polarities[category].append(polarity)

        # Compute statistics
        stats = {}
        for category, polarities in category_polarities.items():
            if len(polarities) > 0:
                stats[category] = {
                    "mean": sum(polarities) / len(polarities),
                    "count": len(polarities),
                    "negative_count": sum(1 for p in polarities if p < 0),
                    "positive_count": sum(1 for p in polarities if p > 0),
                    "neutral_count": sum(1 for p in polarities if p == 0),
                }
            else:
                stats[category] = {
                    "mean": 0.0,
                    "count": 0,
                    "negative_count": 0,
                    "positive_count": 0,
                    "neutral_count": 0,
                }

        return stats
