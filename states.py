# pylint: disable=all
from FeatureCloud.app.engine.app import AppState, app_state, Role
from typing import List
from src.client import FedHistRandomForestClient
from logic import main

# if missing values want to be supported, check the MISSING_VALUES_SUPPORT comments
# here and in the called classes/functions
@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    The only state used
    Trains a random forest in a federated manner.
    All clients train the same trees together. Binning is used.
    """

    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        main(self)
        return 'terminal'
