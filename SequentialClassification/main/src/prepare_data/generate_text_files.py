"""Creates all text files needed to train/test HMMs with HTK, including
wordList, dict, grammar, and all_labels.mlf.

Methods
-------
generate_text_files
_get_unique_words
_write_grammar_line
_generate_word_list
_generate_word_dict
_generate_grammar
_generate_mlf_file
"""
import os
import glob
from io import TextIOWrapper


def generate_text_files(features_dir: str) -> None:
    """Creates all text files needed to train/test HMMs with HTK,
    including wordList, dict, grammar, and all_labels.mlf.
    
    Parameters
    ----------
    features_dir : str
        Unix style pathname pattern pointing to all the features
        extracted from training data.
    """

    unique_words = _get_unique_words(features_dir)

    _generate_word_list(unique_words)

    _generate_word_dict(unique_words)

    _generate_grammar(features_dir)

    _generate_mlf_file()


def _get_unique_words(features_dir: str) -> set:
    """Gets all unique words from a data set.

    Parameters
    ----------
    features_dir : str
        Unix style pathname pattern pointing to all the features
        extracted from training data.

    Returns
    -------
    unique_words : set
        Set of all words found in the training data.
    """

    unique_words = set()
    features_filepaths = glob.glob(os.path.join(features_dir, '**/*.data'), recursive = True)
    features_filepaths.extend(glob.glob(os.path.join(features_dir, '**/*.json'), recursive = True))
    split_index = 1

    for features_filepath in features_filepaths:
        filename = features_filepath.split('/')[-1]
        phrase = filename.split('.')[split_index].split('_')
        phrase = [word.lower() for word in phrase]
        unique_words = unique_words.union(phrase)

    unique_words = sorted(unique_words)

    return unique_words


def _generate_word_list(unique_words: list) -> None:
    """Generates wordList file containing all unique words and silences.

    Parameters
    ----------
    unique_words : set
        Set of all words found in the training data.
    """
    
    word_list = list(unique_words)
    word_list += ['sil0', 'sil1']

    with open('wordList', 'w') as f:
        
        for word in word_list[:-1]:
            f.write('{}\n'.format(word))
        f.write('{}'.format(word_list[-1]))


def _generate_word_dict(unique_words: list) -> None:
    """Generates dict file containing key-value pairs of words. In our
    case, the key and value are both the single, unique word.

    Parameters
    ----------
    unique_words : set
        Set of all words found in the training data.
    """
    
    word_list = list(unique_words)
    word_list += ['sil0', 'sil1']

    with open('dict', 'w') as f:

        f.write('SENT-START [] sil0\n')
        f.write('SENT-END [] sil1\n')
        
        for word in word_list[:-1]:
            f.write('{} {}\n'.format(word, word))
        f.write('{} {}\n'.format(word_list[-1], word_list[-1]))


def _write_grammar_line(
        f: TextIOWrapper, part_of_speech: str, words: list, n='') -> None:
    """Writes a single line to grammar.txt.

    Parameters
    ----------
    f : TextIOWrapper
        Buffered text stream to write to grammar.txt file.

    part_of_speech : str
        Part of speech being written on line.

    words : list
        List of words to be written to line.

    n : str, optional, by default ''
        If a part of speech can be included more than once in the
        grammar, each one should have a distinct count.
    """

    f.write('${}{} = '.format(part_of_speech, n))
    for word in words[:-1]:
        f.write('{} | '.format(word))
    f.write('{};\n'.format(words[-1]))


def _generate_grammar(features_dir: str) -> None:
    """Creates rule-based grammar depending on the length of the longest
    phrase of the dataset.

    Parameters
    ----------
    features_dir : str
        Unix style pathname pattern pointing to all the features
        extracted from training data.
    """

    subjects = set()
    prepositions = set()
    objects = set()
    adjectives = set()
    max_phrase_len = 0
    features_filepaths = glob.glob(os.path.join(features_dir, '**/*.data'), recursive = True)
    features_filepaths.extend(glob.glob(os.path.join(features_dir, '**/*.json'), recursive = True))
    split_index = 1

    for features_filepath in features_filepaths:
        filename = features_filepath.split('/')[-1]
        phrase = filename.split('.')[split_index].split('_')
        phrase = [word.lower() for word in phrase]
        phrase_len = len(phrase)
        max_phrase_len = max(phrase_len, max_phrase_len)

        if phrase_len == 3:

            subject, preposition, object_ = phrase
            subjects.add(subject)
            prepositions.add(preposition)
            objects.add(object_)

        elif phrase_len == 4:

            subject, preposition, adjective, object_ = phrase
            subjects.add(subject)
            prepositions.add(preposition)
            adjectives.add(adjective)
            objects.add(object_)

        elif phrase_len == 5:

            adjective_1, subject, preposition, adjective_2, object_ = phrase
            adjectives.add(adjective_1)
            subjects.add(subject)
            prepositions.add(preposition)
            adjectives.add(adjective_2)
            objects.add(object_)

    subjects = list(subjects)
    prepositions = list(prepositions)
    objects = list(objects)
    adjectives = list(adjectives)

    with open('grammar.txt', 'w') as f:
    
        if max_phrase_len == 3:

            _write_grammar_line(f, 'subject', subjects)
            _write_grammar_line(f, 'preposition', prepositions)
            _write_grammar_line(f, 'object', objects)
            f.write('\n')
            f.write('(SENT-START $subject $preposition $object SENT-END)')
            f.write('\n')
            
        elif max_phrase_len == 4:

           _write_grammar_line(f, 'subject', subjects)
           _write_grammar_line(f, 'preposition', prepositions)
           _write_grammar_line(f, 'adjective', adjectives)
           _write_grammar_line(f, 'object', objects)
           f.write('\n')
           f.write('(SENT-START $subject $preposition [$adjective] $object SENT-END)')
           f.write('\n')

        elif max_phrase_len == 5:

            _write_grammar_line(f, 'adjective', adjectives, 1)
            _write_grammar_line(f, 'subject', subjects)
            _write_grammar_line(f, 'preposition', prepositions)
            _write_grammar_line(f, 'adjective', adjectives, 2)
            _write_grammar_line(f, 'object', objects)
            f.write('\n')
            f.write('(SENT-START [$adjective1] $subject $preposition [$adjective2] $object SENT-END)')
            f.write('\n')

    f.close()


def _generate_mlf_file() -> None:
    """Creates all_labels.mlf file that contains every phrase in the 
    dataset.
    """

    htk_filepaths = os.path.join('data', 'htk', '*.htk')
    filenames = glob.glob(htk_filepaths)

    with open('all_labels.mlf', 'w') as f:
        
        f.write('#!MLF!#\n')

        for filename in filenames:

            label = filename.split('/')[-1].replace('htk', 'lab')
            phrase = label.split('.')[1].split('_')

            f.write('"*/{}"\n'.format(label))
            f.write('sil0\n')

            for word in phrase:

                f.write('{}\n'.format(word.lower()))

            f.write('sil1\n')
            f.write('.\n')