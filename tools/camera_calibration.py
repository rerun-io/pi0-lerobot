import tyro

from pi0_lerobot.apis.camera_calibration import CameraCalibConfig, calibrate_camera

if __name__ == "__main__":
    calibrate_camera(tyro.cli(CameraCalibConfig))
