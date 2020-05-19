"""Parses hresults file and returns them as dict.

Methods
-------
get_results
"""
import os

def get_results(results_file: str) -> dict:
    """Parses hresults file and returns them as dict.

    Parameters
    ----------
    results_file : str
        File path of hresults file to be parsed. Generally the last in
        a group of trained models.

    Returns
    -------
    results_dict : dict
        Results containing substitutions, deletions, insertions, error,
        and sentence error.
    """

    results_dict = {}
    confusion_string = ""

    with open(results_file, 'r') as lf:
        start_file = False
        for line in lf:
            l = line.rstrip()
            if l == "    ,-------------------------------------------------------------.":
                start_file = True
            if start_file:
                confusion_string += l + "\n"
                if "| Sum/Avg |" in l:
                    vals = l.split("|")[3].strip().split()
                    results_dict['substitutions'] = float(vals[1])
                    results_dict['deletions'] = float(vals[2])
                    results_dict['insertions'] = float(vals[3])
                    results_dict['error'] = float(vals[4])
                    results_dict['sentence_error'] = float(vals[5])
                    
        results_dict['confusion_matrix'] = confusion_string

    return results_dict
