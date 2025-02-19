"""
A class to use to simulate a federated learning environment locally. This class
adheres to the ProtocolFedLearning protocol of this project. This class represents
a single client in a federated learning environment.
- SharedDictionary: A helper class that allows multiple instances to share a single dictionary concurrently.
- LocalFedLearningSimulator: The simulator class, representing a single client in a federated learning environment.
- LocalFedLearningSimulationWrapper: A wrapper class to simulate a federated learning environment locally. Contains
    multiple instances of LocalFedLearningSimulator and prepares the folder for running them.
"""
from threading import Lock
import os
import time
import shutil
from typing import Any, Optional, List
from .protocolfedlearningclass import ProtocolFedLearning

WAITING_TIME = 0.1

class SharedDictionary:
    """
    A helper class that allows multiple instances to share a single dictionary concurrently.
    """

    def __init__(self, num_clients: int):
        self._shared_dict = {}
        self._lock = Lock()
        self.num_clients = num_clients

    def get(self, key):
        """
        Gets the value associated with the given key from the shared dictionary.
        """
        with self._lock:
            return self._shared_dict.get(key)

    def get_all_non_global_values(self):
        """
        Gets all key-value pairs from the shared dictionary.
        """
        with self._lock:
            result = []
            for key, value in self._shared_dict.items():
                if key not in ['global', 'times_accesed_global']:
                    result.append(value)
            return result

    def delete_all_but_global(self):
        """
        Deletes all key-value pairs from the shared dictionary except the one with key 'global'.
        """
        with self._lock:
            if 'global' in self._shared_dict and 'times_accesed_global' in self._shared_dict:
                self._shared_dict = {'global': self._shared_dict['global'],
                                     'times_accesed_global': self._shared_dict['times_accesed_global']}
            else:
                self._shared_dict = {}

    def increment_times_accesed_global(self):
        """
        Increments the value associated with the key 'times_accesed_global' in the shared dictionary.
        """
        with self._lock:
            if 'times_accesed_global' in self._shared_dict:
                self._shared_dict['times_accesed_global'] += 1
            else:
                raise ValueError("The key 'times_accesed_global' does not exist in the shared dictionary, but trying to increment it")
            # cleanup 'global' if all clients have accessed it
            if self._shared_dict['times_accesed_global'] == self.num_clients:
                # remove the global data
                del self._shared_dict['global']
                del self._shared_dict['times_accesed_global']

    def update_global(self, data):
        """
        Updates the value associated with the key 'global' in the shared dictionary.
        """
        with self._lock:
            self._shared_dict['global'] = data
            self._shared_dict['times_accesed_global'] = 0
                # reset as this new data has not been accessed by any client yet

    def set(self, key, value):
        """
        Sets the value associated with the given key in the shared dictionary.
        """
        with self._lock:
            self._shared_dict[key] = value

    def delete(self, key):
        """
        Removes the key-value pair from the shared dictionary.
        """
        with self._lock:
            if key in self._shared_dict:
                del self._shared_dict[key]

class LocalFedLearningSimulator(ProtocolFedLearning):
    """
    The simulator class, representing a single client in a federated learning environment.
    """
    def __init__(self,
                 is_coordinator: bool,
                 client_id: int,
                 num_clients: int,
                 inputfolder: str,
                 outputfolder: str,
                 shared_dict: SharedDictionary):
        """
        The constructor, the following parameters are required:

        Args:
            is_coordinator: Boolean variable, if True the this client
                represents the coordinator. False otherwise.
            client_id: The id of the client
            num_clients: The total number of clients in the federated learning
                environment. This is needed to correctly gather data,
                specifically to know how many data packets to wait for
            inputfolder: The path to the input folder containing the data for this client
            outputfolder: The path to the output folder to store the results
            shared_dict: A shared dictionary to store the data.
                The other clients (instances of this class) receive the same
                dictionary. The client uses it's client_id as key to access the data.
        """
        self.coordinator = is_coordinator
        self.client_id = client_id
        if client_id == 'global':
            raise ValueError("The client_id 'global' is reserved for global data")
        self.num_clients = num_clients
        self.inputfolder = inputfolder
        self.outputfolder = outputfolder
        self.shared_dict = shared_dict
        self.previously_awaited_data = None

    @property
    def is_coordinator(self):
        return self.coordinator

    def send_data_to_coordinator(self,
                                 data,
                                 send_to_self=True,
                                 use_smpc=False,
                                 use_dp=False,
                                 memo=None):
        print(f"Client {self.client_id} sending data to coordinator")
        while True:
            # we can only send if what we send before was gather already
            # the gathering deletes the data from the shared dictionary
            # we therefore only need to make sure that get returns None
            if self.shared_dict.get(self.client_id) is None:
                self.shared_dict.set(self.client_id, data)
                break
            time.sleep(WAITING_TIME)

    def gather_data(self,
                    is_json: bool=False,
                    use_smpc: bool=False,
                    use_dp: bool=False,
                    memo: Optional[Any]=None):
        if not self.coordinator:
            raise ValueError("Only the coordinator can gather data")
        # wait for enough data to arrive in a loop
        while True:
            data_packets = self.shared_dict.get_all_non_global_values()
            print(f"gathering data, got {len(data_packets)} of {self.num_clients} data packets")
            if len(data_packets) == self.num_clients:
                # reset the shared dictionary
                self.shared_dict.delete_all_but_global()
                return data_packets
            time.sleep(WAITING_TIME)

    def broadcast_data(self,
                       data: Any,
                       send_to_self: bool = True,
                       use_dp: bool = False,
                       memo: Optional[Any] = None) -> None:
        if not self.coordinator:
            raise ValueError("Only the coordinator can broadcast data")
        print(f"Broadcasting data to all clients")
        while True:
            if not self.shared_dict.get('times_accesed_global'):
                # only happens after all clients have accessed the global data
                # or at the start when no client has accessed the global data
                self.shared_dict.update_global(data)
                break
            time.sleep(WAITING_TIME)

    def await_data(self,
                     n: int = 1,
                     unwrap: bool = True,
                     is_json: bool = False,
                     use_dp: bool = False,
                     use_smpc: bool = False,
                     memo: Optional[Any] = None):

        if n != 1:
            raise ValueError("This method only supports n=1 right now")
        while True:
            data = self.shared_dict.get('global')

            if data is not None and data != self.previously_awaited_data:
                # data is not None -> some data was broadcasted
                # data != self.previously_awaited_data -> if this is true other clients have not yet
                # accessed the data. We need to wait until all clients have accessed the data
                print(f"Client {self.client_id} received data, incrementing times_accesed_global")
                self.shared_dict.increment_times_accesed_global()
                self.previously_awaited_data = data
                break
            time.sleep(WAITING_TIME)

        # unwrap is not needed as we never wrap the data in the first place
        return data

class LocalFedLearningSimulationWrapper:
    """
    A wrapper class to simulate a federated learning environment locally.
    This class is used to create multiple instances of LocalFedLearningSimulator
    and run them concurrently.
    """
    def __init__(self,
                 clientfolders: List[str],
                 outputfolders: List[str],
                 generic_dir: Optional[str]) -> None:
        """
        The following arguments are required:

        Args:
            clientfolders: A list of paths to the client folders containing the data
            generic_dir: The path to the generic directory. The content of this folder is copied to
                each client folder. If the file in the generic folder already exists in the
                client folder, it is not copied. Folders in the generic folder are not copied.
                The first clientfolder is used as the coordinator.
        """
        # checks
        if len(clientfolders) < 2:
            raise ValueError("At least two client folders are required")
        if len(clientfolders) != len(outputfolders):
            raise ValueError("The number of client folders and output folders must be the same")
        # basic variables
        self.num_clients = len(clientfolders)
        self.shared_dict = SharedDictionary(num_clients=self.num_clients)
        self.created_files = []

        # copy files from the generic folder to each client folder
        for clientfolder in clientfolders:
            # copy files from the generic folder to the client folder
            if generic_dir is not None:
                for root, _, files in os.walk(generic_dir):
                    for file in files:
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(clientfolder, file)
                        if os.path.exists(dst_file):
                            if os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                                print("WARNING: File from generic folder already exists in client " +\
                                    "but generic file is newer. Overwriting")
                                shutil.copy(src_file, dst_file)
                                self.created_files.append(dst_file)
                            else:
                                print("WARNING: File from generic folder already exists in client " +\
                                    "and is newer. Skipping copy of the file from generic folder")
                        else:
                            # file doesn't exist yet, copy it
                            shutil.copy(src_file, dst_file)
                            self.created_files.append(dst_file)

        # create the client instances
        self.clients: List[LocalFedLearningSimulator] = []
        for i, clientfolder in enumerate(clientfolders):
            is_coordinator = i == 0
            client = LocalFedLearningSimulator(is_coordinator=is_coordinator,
                                               client_id=i,
                                               num_clients=self.num_clients,
                                               inputfolder=clientfolder,
                                               outputfolder=outputfolders[i],
                                               shared_dict=self.shared_dict)
            self.clients.append(client)

    def cleanup_created_files(self):
        """
        Removes any files that were added to client folders during the init process
        from the generic folder.
        """
        for file in self.created_files:
            os.remove(file)
        self.created_files = []
