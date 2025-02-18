import tyro

from pi0_lerobot.apis.visualize_assembly import VisualzeConfig, visualize_data

if __name__ == "__main__":
    visualize_data(tyro.cli(VisualzeConfig))
