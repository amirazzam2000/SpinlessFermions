import argparse


def add_bool_arg(parser, name, short, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-' + short,'--' + name, dest=name, action='store_true')
    group.add_argument('-no-' + short ,'--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name:default})



parser = argparse.ArgumentParser(prog="SpinlessFermions",
                                 usage='%(prog)s [options]',
                                 description="A Neural Quantum State (NQS) solution to one-dimensional fermions interacting in a Harmonic trap",
                                 epilog="and fin")

add_bool_arg(parser, 'freeze', 'F')
args = parser.parse_args()

print(args.freeze)