"""Save results of training run to JSON file for post-processing.

Methods
-------
save_results
"""
import os

from .json_data import load_json, dump_json


def save_results(results_dict: dict, results_file: str, action: str = 'w') -> None:
    """Save results of training run to JSON file for post-processing.
    Will auto-increment the last index to append to an existing JSON.

    Parameters
    ----------
    results_dict : dict
        Results from training run to be saved.

    results_file : str
        File path at which to save results.
    """

    if os.path.exists(results_file):

        results = load_json(results_file)
        results[len(results) + 1] = results_dict

    else:

        results = {1: results_dict}
        
    dump_json(results_file, results, action)
