"""Initializes models for words in training set from a given prototype.
Should be called after generate_prototype and in conjuncion with
configs/prototypes.json.

Methods
-------
initialize_models
"""
import os


def initialize_models(
        prototype_filepath: str, words: list, hmm_dir: str) -> None:
    """Initializes models for words in training set from a given
    prototype. Should be called after generate_prototype and in 
    conjuncion with configs/prototypes.json.

    Parameters
    ----------
    prototype_filepath : str
        File path prototype to be used to initialize model.

    words : list
        All words to be initalized with given prototype.

    hmm_dir : str
        Directory at which to save newly created models.
    """

    with open(prototype_filepath, 'r') as f:
        prototype = f.read().strip('\r\n')

    for word in words:

        word = word.strip('\r\n')
        hmm_word_filepath = os.path.join(hmm_dir, word)

        with open(hmm_word_filepath, 'w') as f:
            f.write(prototype.replace('prototype', word))
