from rtmlib.visualization import coco133

COCO_133_IDS: list[int] = [id for id, kpt_info in coco133["keypoint_info"].items()]
link_dict = {}
for _, kpt_info in coco133["keypoint_info"].items():
    link_dict[kpt_info["name"]] = kpt_info["id"]

COCO_133_LINKS: list[tuple[int, int]] = []
for _, ske_info in coco133["skeleton_info"].items():
    link = ske_info["link"]
    COCO_133_LINKS.append((link_dict[link[0]], link_dict[link[1]]))

COCO_133_ID2NAME: dict[int, str] = {id: kpt_info["name"] for id, kpt_info in coco133["keypoint_info"].items()}
