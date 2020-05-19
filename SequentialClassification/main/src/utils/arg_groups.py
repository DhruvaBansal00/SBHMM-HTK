"""Splits argument groups into separate Namespaces that can be passed
to functions individually.

Methods
-------
get_arg_groups
"""
from argparse import ArgumentParser, Namespace


def get_arg_groups(parser: ArgumentParser, args: Namespace) -> dict:
    """Splits argument groups into separate Namespaces that can be 
    passed to functions individually.

    Parameters
    ----------
    parser : ArgumentParser
        Parser containing all arguments.

    args : Namespace
        Args after having been parsed.

    Returns
    -------
    arg_groups : dict
        Argument groups separated out by Namespace.
    """

    arg_groups={}

    for group in parser._action_groups:
        group_dict = {
            a.dest:getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = Namespace(**group_dict)

    return arg_groups
