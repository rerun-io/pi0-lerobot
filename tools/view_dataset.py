import tyro
from pi0_aloha.apis.visualize_dataset_rerun import ViewDatasetArgs, new_main

if __name__ == "__main__":
    new_main(tyro.cli(ViewDatasetArgs))
