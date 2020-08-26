"""Generates prototype files used to initalize models. Should be used
with configs/prototypes.json. Different words can be initialized with
different prototypes.

Methods
-------
generate_prototype
"""
import sys


def generate_prototype(n_states: int, n_features: int, output_filepath: str, 
                       mean: float = 0.0, variance: float = 1.0, 
                       transition_prob: float = 0.6) -> None:
    """Generates prototype files used to initalize models.

    Parameters
    ----------
    n_states : int
        Number of states each model has.

    n_features : int
        Number of features being used to train each model.

    output_filepath : str
        File path at which to save prototype.

    mean : float, optional, by default 0.0
        Initial value to use as mean of all features.

    variance : float, optional, by default 1.0
        Initial value to use as variance of all features.

    transition_prob : float, optional, by default 0.6
        Initial probability of transition from one state to the next.
    """

    with open(output_filepath, 'w') as f:

        f.write('~o\n')
        f.write('<VecSize> {} <USER>\n'.format(n_features))
        f.write('~h "prototype"\n')
        f.write('<BeginHMM>\n')
        f.write('<NumStates> {}\n'.format(n_states))

        for i in range(2, n_states):

            f.write('<State> {}\n'.format(i))
            f.write('<Mean> {}\n'.format(n_features))
            f.write(' '.join([str(mean)]*n_features) + '\n')
            f.write('<Variance> {}\n'.format(n_features))
            f.write(' '.join([str(variance)]*n_features) + '\n')

        f.write('<TransP> {}\n'.format(n_states))
        row = ['0.0'] + ['1.0'] + ['0.0']*(n_states - 2)
        f.write(' '.join(row) + '\n')

        for i in range(1, n_states-2):
            # if i == 6:
            #     row = ['0.0']*i + [str(0.9*transition_prob), str(0.9-0.9*transition_prob)] + ['0.0']*3 + ['0.1'] + ['0.0']*(n_states - i - 6)
            #     f.write(' '.join(row) + '\n')
            # else:
            row = ['0.0']*i + [str(transition_prob), str(1-transition_prob)] + ['0.0']*(n_states - i - 2)
            f.write(' '.join(row) + '\n')
            # row = ['0.0']*i + [str(0.8*transition_prob), str((1-transition_prob)*0.8)] + [str(0.2/(n_states - i - 2.0))]*(n_states - i - 2)
            # f.write(' '.join(row) + '\n')
            # row = ['0.0']*i + [str(transition_prob), str(1-transition_prob)] + ['0.0']*(n_states - i - 2)
            # f.write(' '.join(row) + '\n')

        row = ['0.0']*(n_states - 2) + [str(transition_prob), str(1-transition_prob)]
        f.write(' '.join(row) + '\n')

        f.write(' '.join(['0.0']*n_states) + '\n')
        f.write('<EndHMM>\n')
