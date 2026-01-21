"""
Utilities for analyzing diffusion simulation results.

Provides helper functions for running tracked simulations and analyzing
the diffusion sequence according to Rogers' theory.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx
from ..core.schemas import NetworkConfig, SimulationConfig, SimulationResult
from ..networks import SmallWorldNetworkBuilder
from ..population import PopulationGenerator
from ..decisions.tpb import TPBDecisionModel
from ..decisions.components import FrictionModel
from ..simulation.tracked_engine import TrackedTPBDiffusionEngine
from ..agents import TPBAgent


def run_tracked_simulation(
    n: int = 200,
    k: int = 6,
    p: float = 0.1,
    seed: int = 42,
    initial_adopters: int = 5,
    initial_strategy: str = "innovators",
    w_attitude: float = 0.4,
    w_social_norm: float = 0.35,
    w_pbc: float = 0.25,
    friction: float = 0.0,
    max_steps: int = 100,
) -> Tuple[SimulationResult, Dict, nx.Graph]:
    """
    Run a TPB simulation with detailed tracking of diffusion process.

    This function creates a complete simulation setup and tracks:
    - When each agent becomes aware
    - When each agent adopts
    - Who informed each agent
    - Influence depth from seed adopters

    Args:
        n: Number of nodes in network
        k: Average degree (must be even)
        p: Rewiring probability for small-world network
        seed: Random seed for reproducibility
        initial_adopters: Number of seed adopters
        initial_strategy: Strategy for selecting seeds ("innovators", "random", "high_degree")
        w_attitude: TPB weight for attitude
        w_social_norm: TPB weight for social norm
        w_pbc: TPB weight for perceived behavioral control
        friction: Friction level (0-1)
        max_steps: Maximum simulation steps

    Returns:
        Tuple of (SimulationResult, tracking_data, graph)
        - SimulationResult: Standard simulation results
        - tracking_data: Dict with 'agent_info' and 'influence_events'
        - graph: The network graph (for visualization)
    """
    # Build network
    network_config = NetworkConfig(n=n, k=k, p=p, seed=seed)
    builder = SmallWorldNetworkBuilder()
    graph = builder.build(network_config)

    # Generate heterogeneous population
    pop_gen = PopulationGenerator(seed=seed)
    pop_gen.generate_population(graph)

    # Select initial adopters
    initial_ids = pop_gen.select_initial_adopters(
        graph, initial_adopters, strategy=initial_strategy
    )

    # Create TPB decision model
    friction_model = FrictionModel(base_friction=friction, mode="fixed")
    decision_model = TPBDecisionModel(
        w_attitude=w_attitude,
        w_social_norm=w_social_norm,
        w_pbc=w_pbc,
        friction=friction_model,
        sn_strategy="mean",
        intention_mode="sigmoid",
        share_threshold=0.5,
    )

    # Create simulation config
    sim_config = SimulationConfig(max_steps=max_steps, seed=seed, initial_adopters=initial_adopters)

    # Run tracked simulation
    engine = TrackedTPBDiffusionEngine(graph, sim_config, network_config, decision_model)
    engine.initialize(initial_adopter_ids=initial_ids)
    result, tracking_data = engine.run()

    return result, tracking_data, graph


def analyze_diffusion_sequence(tracking_data: Dict) -> Dict:
    """
    Analyze diffusion sequence statistics by Rogers categories.

    Args:
        tracking_data: Tracking data from run_tracked_simulation

    Returns:
        Dictionary with statistics for each category
    """
    agent_info = tracking_data['agent_info']
    categories = ['innovator', 'early_adopter', 'early_majority', 'late_majority', 'laggard']

    stats = {}
    for category in categories:
        # Find all agents in this category
        agents_in_cat = [
            aid for aid, info in agent_info.items()
            if info['category'] == category
        ]

        # Collect awareness times
        aware_times = [
            info['aware_at'] for aid, info in agent_info.items()
            if info['category'] == category and info['aware_at'] is not None
        ]

        # Collect adoption times
        adopted_times = [
            info['adopted_at'] for aid, info in agent_info.items()
            if info['category'] == category and info['adopted_at'] is not None
        ]

        # Calculate statistics
        stats[category] = {
            'count': len(agents_in_cat),
            'aware_count': len(aware_times),
            'aware_rate': len(aware_times) / len(agents_in_cat) if agents_in_cat else 0,
            'first_aware': min(aware_times) if aware_times else None,
            'median_aware': float(np.median(aware_times)) if aware_times else None,
            'mean_aware': float(np.mean(aware_times)) if aware_times else None,
            'last_aware': max(aware_times) if aware_times else None,
            'adopted_count': len(adopted_times),
            'adoption_rate': len(adopted_times) / len(agents_in_cat) if agents_in_cat else 0,
            'first_adopted': min(adopted_times) if adopted_times else None,
            'median_adopted': float(np.median(adopted_times)) if adopted_times else None,
            'mean_adopted': float(np.mean(adopted_times)) if adopted_times else None,
            # Calculate aware to adopt delay
            'aware_adopt_delays': [
                info['adopted_at'] - info['aware_at']
                for aid, info in agent_info.items()
                if (info['category'] == category and
                    info['aware_at'] is not None and
                    info['adopted_at'] is not None)
            ]
        }

        # Add delay statistics
        delays = stats[category]['aware_adopt_delays']
        if delays:
            stats[category]['mean_delay'] = float(np.mean(delays))
            stats[category]['median_delay'] = float(np.median(delays))
        else:
            stats[category]['mean_delay'] = None
            stats[category]['median_delay'] = None

    return stats


def check_rogers_sequence(stats: Dict) -> Dict:
    """
    Verify if diffusion sequence follows Rogers' theory expectations.

    Args:
        stats: Statistics from analyze_diffusion_sequence

    Returns:
        Dictionary with validation results
    """
    categories = ['innovator', 'early_adopter', 'early_majority', 'late_majority', 'laggard']

    # Check awareness timing order
    first_aware_times = [
        (cat, stats[cat]['first_aware'])
        for cat in categories
        if stats[cat]['first_aware'] is not None
    ]
    first_aware_times.sort(key=lambda x: x[1])

    # Check adoption timing order
    first_adopted_times = [
        (cat, stats[cat]['first_adopted'])
        for cat in categories
        if stats[cat]['first_adopted'] is not None
    ]
    first_adopted_times.sort(key=lambda x: x[1])

    # Expected order
    expected_order = [c for c in categories]

    # Check if actual order matches expected
    aware_order = [cat for cat, _ in first_aware_times]
    adopted_order = [cat for cat, _ in first_adopted_times]

    # Filter expected order to only include categories that appear
    expected_aware = [c for c in expected_order if c in aware_order]
    expected_adopted = [c for c in expected_order if c in adopted_order]

    validation = {
        'aware_sequence': aware_order,
        'expected_aware_sequence': expected_aware,
        'aware_matches_theory': aware_order == expected_aware,
        'adopted_sequence': adopted_order,
        'expected_adopted_sequence': expected_adopted,
        'adopted_matches_theory': adopted_order == expected_adopted,
        'first_aware_times': dict(first_aware_times),
        'first_adopted_times': dict(first_adopted_times),
    }

    return validation


def print_diffusion_report(stats: Dict, validation: Optional[Dict] = None):
    """
    Print a formatted report of diffusion statistics.

    Args:
        stats: Statistics from analyze_diffusion_sequence
        validation: Optional validation results from check_rogers_sequence
    """
    print("\n" + "="*90)
    print("创新扩散时序分析报告")
    print("="*90)
    print(f"{'分类':<20} {'人数':<8} {'Aware率':<10} {'首次Aware':<12} {'中位Aware':<12} {'Adoption率':<12}")
    print("-"*90)

    for cat, data in stats.items():
        first = f"{int(data['first_aware'])}" if data['first_aware'] is not None else "N/A"
        median = f"{data['median_aware']:.1f}" if data['median_aware'] is not None else "N/A"
        print(f"{cat.replace('_', ' ').title():<20} {data['count']:>6}  "
              f"{data['aware_rate']*100:>8.1f}%  {first:>11}  {median:>11}  "
              f"{data['adoption_rate']*100:>10.1f}%")

    # Print delay statistics
    print("\n" + "="*90)
    print("Aware → Adopt 延迟统计（反映决策速度）")
    print("="*90)
    print(f"{'分类':<20} {'平均延迟':<15} {'中位延迟':<15}")
    print("-"*90)

    for cat, data in stats.items():
        mean_delay = f"{data['mean_delay']:.1f}" if data['mean_delay'] is not None else "N/A"
        median_delay = f"{data['median_delay']:.1f}" if data['median_delay'] is not None else "N/A"
        print(f"{cat.replace('_', ' ').title():<20} {mean_delay:>13}  {median_delay:>13}")

    # Print validation results
    if validation:
        print("\n" + "="*90)
        print("Rogers理论验证")
        print("="*90)

        print(f"\n实际Awareness顺序: {' → '.join([c.replace('_', ' ').title() for c in validation['aware_sequence']])}")
        print(f"预期顺序: {' → '.join([c.replace('_', ' ').title() for c in validation['expected_aware_sequence']])}")

        if validation['aware_matches_theory']:
            print("✅ Awareness顺序符合Rogers理论预期！")
        else:
            print("⚠️  Awareness顺序与理论预期有偏差")

        print(f"\n实际Adoption顺序: {' → '.join([c.replace('_', ' ').title() for c in validation['adopted_sequence']])}")
        print(f"预期顺序: {' → '.join([c.replace('_', ' ').title() for c in validation['expected_adopted_sequence']])}")

        if validation['adopted_matches_theory']:
            print("✅ Adoption顺序符合Rogers理论预期！")
        else:
            print("⚠️  Adoption顺序与理论预期有偏差")

    print("\n" + "="*90)


def get_category_timeseries(tracking_data: Dict, result: SimulationResult) -> Dict:
    """
    Get awareness and adoption time series broken down by Rogers categories.

    Args:
        tracking_data: Tracking data from simulation
        result: Simulation result object

    Returns:
        Dictionary mapping category to time series data
    """
    agent_info = tracking_data['agent_info']
    categories = ['innovator', 'early_adopter', 'early_majority', 'late_majority', 'laggard']

    timeseries = {}
    max_step = result.total_steps

    for category in categories:
        # Get all agents in this category
        agents_in_cat = [
            aid for aid, info in agent_info.items()
            if info['category'] == category
        ]

        # Track cumulative aware and adopted counts over time
        aware_count = []
        adopted_count = []

        for step in range(max_step + 1):
            aware_at_step = sum(
                1 for aid in agents_in_cat
                if agent_info[aid]['aware_at'] is not None
                and agent_info[aid]['aware_at'] <= step
            )
            adopted_at_step = sum(
                1 for aid in agents_in_cat
                if agent_info[aid]['adopted_at'] is not None
                and agent_info[aid]['adopted_at'] <= step
            )

            aware_count.append(aware_at_step)
            adopted_count.append(adopted_at_step)

        timeseries[category] = {
            'steps': list(range(max_step + 1)),
            'aware_count': aware_count,
            'adopted_count': adopted_count,
            'total_in_category': len(agents_in_cat),
        }

    return timeseries
