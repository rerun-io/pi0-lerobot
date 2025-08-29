import tyro
from wilor_nano.api.wilor_inference import WilorConfig, main

if __name__ == "__main__":
    main(tyro.cli(WilorConfig))
