from typing import Protocol

class ProtocolFedLearning(Protocol):
    """
    This protocol defines the interface for a federated learning library to
    be used with the federated histogram based random forest of this project.
    This protocol is compatible with FeatureCloud's AppState class.
    However, by implementing this protocol, any other federated learning
    library can be used for the federated histogram based random forest.
    """
