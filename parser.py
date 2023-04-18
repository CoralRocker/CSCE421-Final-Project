import argparse


def parse():
    parser = argparse.ArgumentParser("tabular_training")

    # Data
    parser.add_argument(
        "--data_path", type=str, default=".", help="path to your data directory"
    )

    ############################ Your Code Here ############################
    # You can add more arguments like the one above in this space

    ########################################################################

    return parser.parse_args()
