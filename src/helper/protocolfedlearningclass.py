"""
A protocol defining the federated learning class to be used by the federated
histogram based random forest. This protocol is compatible with FeatureCloud's
AppState class.
"""
from typing import Protocol, Union, List, Any, Optional

class ProtocolFedLearning(Protocol):
    """
    This protocol defines the interface for a federated learning library to
    be used with the federated histogram based random forest of this project.
    This protocol is compatible with FeatureCloud's AppState class.
    However, by implementing this protocol, any other federated learning
    library can be used for the federated histogram based random forest.
    The base idea is to use star shaped client-coordinator federated learning.
    The coordinator is also a client.
    """
    @property
    def is_coordinator(self):
        """ Boolean variable, if True the this AppState instance represents the
        coordinator. False otherwise.

        """

    def send_data_to_coordinator(self,
                                 data: Any,
                                 send_to_self: bool=True,
                                 use_smpc: bool=False,
                                 use_dp: bool=False,
                                 memo: Optional[Any]=None) -> None:
        """
        Sends the given data to the coordinator
        """

    def gather_data(self,
                    is_json: bool=False,
                    use_smpc: bool=False,
                    use_dp: bool=False,
                    memo: Optional[str]=None) -> Union[Any, List[Any]]:
        """
        Receives the data from the clients which used send_data_to_coordinator
        Waits for ALL clients to send data before returning the data.
        """

    def broadcast_data(self,
                       data: Any,
                       send_to_self: bool = True,
                       use_dp: bool = False,
                       memo: Optional[Any] = None) -> None:
        """
        Sends data from the coordinator to all clients. Used to share global
        aggregations.
        """

    def await_data(self,
                   n: int = 1,
                   unwrap: bool = True,
                   is_json: bool = False,
                   use_dp: bool = False,
                   use_smpc: bool = False,
                   memo: Optional[Any] = None) -> Union[Any, List[Any]]:
        """
        Waits for exactly one data piece. Used to receive data from the
        broadcast_data method by all clients (including the coordinator).
        """
