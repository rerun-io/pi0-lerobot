import tyro

from pi0_lerobot.apis.tesnordict_test import TDArgs, test_td

if __name__ == "__main__":
    test_td(tyro.cli(TDArgs))
