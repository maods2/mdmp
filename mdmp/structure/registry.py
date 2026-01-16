"""
Algorithm registry for structure learning.

This module provides a registry system for learning algorithms, allowing
easy registration and retrieval of algorithms by name.
"""

from typing import Dict, List, Type

from .algorithms import BaseLearningAlgorithm

# Global registry mapping algorithm names to classes
_ALGORITHM_REGISTRY: Dict[str, Type[BaseLearningAlgorithm]] = {}


def register_algorithm(name: str, algorithm_class: Type[BaseLearningAlgorithm]) -> None:
    """
    Register a learning algorithm in the global registry.

    Parameters
    ----------
    name : str
        Name to register the algorithm under (e.g., "hc", "tabu").
    algorithm_class : Type[BaseLearningAlgorithm]
        Algorithm class to register.

    Raises
    ------
    TypeError
        If algorithm_class is not a subclass of BaseLearningAlgorithm.
    ValueError
        If name is already registered.

    Examples
    --------
    >>> from mdmp.structure import register_algorithm, BaseLearningAlgorithm
    >>> class MyAlgorithm(BaseLearningAlgorithm):
    ...     def learn(self, data, nbf, delta, node_names, **kwargs):
    ...         # Implementation
    ...         pass
    >>> register_algorithm("my_algo", MyAlgorithm)
    """
    if not issubclass(algorithm_class, BaseLearningAlgorithm):
        raise TypeError(
            f"algorithm_class must be a subclass of BaseLearningAlgorithm, "
            f"got {type(algorithm_class).__name__}"
        )

    if name in _ALGORITHM_REGISTRY:
        raise ValueError(
            f"Algorithm '{name}' is already registered. "
            f"Use a different name or unregister the existing algorithm first."
        )

    _ALGORITHM_REGISTRY[name] = algorithm_class


def get_algorithm(name: str) -> Type[BaseLearningAlgorithm]:
    """
    Get algorithm class by name from the registry.

    Parameters
    ----------
    name : str
        Name of the algorithm to retrieve.

    Returns
    -------
    Type[BaseLearningAlgorithm]
        Algorithm class.

    Raises
    ------
    ValueError
        If the algorithm name is not registered.

    Examples
    --------
    >>> from mdmp.structure import get_algorithm
    >>> HillClimbingClass = get_algorithm("hc")
    >>> algorithm = HillClimbingClass(verbose=True)
    """
    if name not in _ALGORITHM_REGISTRY:
        available = ", ".join(sorted(_ALGORITHM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown algorithm: '{name}'. "
            f"Available algorithms: {available if available else 'none'}"
        )

    return _ALGORITHM_REGISTRY[name]


def list_algorithms() -> List[str]:
    """
    List all registered algorithm names.

    Returns
    -------
    List[str]
        List of registered algorithm names.

    Examples
    --------
    >>> from mdmp.structure import list_algorithms
    >>> algorithms = list_algorithms()
    >>> print(algorithms)
    ['hc', 'tabu', 'ipa']
    """
    return sorted(_ALGORITHM_REGISTRY.keys())


def unregister_algorithm(name: str) -> None:
    """
    Unregister an algorithm from the registry.

    Parameters
    ----------
    name : str
        Name of the algorithm to unregister.

    Raises
    ------
    ValueError
        If the algorithm name is not registered.

    Examples
    --------
    >>> from mdmp.structure import unregister_algorithm
    >>> unregister_algorithm("my_algo")
    """
    if name not in _ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm '{name}' is not registered.")

    del _ALGORITHM_REGISTRY[name]
