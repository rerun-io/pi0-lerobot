import tyro

from pi0_lerobot.apis.mano_jax_inference import ManoConfig, mano_inference

if __name__ == "__main__":
    mano_inference(tyro.cli(ManoConfig))
