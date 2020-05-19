"""Methods for loading and dumping JSON files. 

Classes
-------
NumpyEncoder

Methods
-------
load_json
dump_json
"""
import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Encoder needed to dump NumPy arrays into a JSON file.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_json(json_file: str) -> dict:
    """Load a JSON file.

    Parameters
    ----------
    json_file : str
        File path of JSON to be loaded.

    Returns
    -------
    data : dict
        Data loaded from JSON.
    """
    
    with open(json_file, 'r') as data_file:
        data = json.loads(data_file.read())
        
    return data


def dump_json(json_file: str, data: dict, action: str = 'w', 
              indent: int = 4) -> None:
    """Dump JSON to file.

    Parameters
    ----------
    json_file : str
        File path of JSON to be dumped.

    data : dict
        Dict containing data to be dumped.

    action : str, optional, by default 'w'
        Action to perform.

    indent : int, optional, by default 4
        Include indentation of this many spaces to improve readability. 
    """
    
    dumps_data = json.dumps(data, indent=indent, cls=NumpyEncoder)
    with open(json_file, action) as data_file:
        print(dumps_data, file=data_file)
