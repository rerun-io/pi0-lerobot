HAND_LINKS = [
    (5, 6),
    (6, 7),
    (7, 0),  # Thumb
    (5, 8),
    (8, 9),
    (9, 10),
    (10, 1),
    (5, 11),  # index
    (11, 12),
    (12, 13),
    (13, 2),
    (5, 14),  # ring
    (14, 15),
    (15, 16),
    (16, 3),
    (5, 17),  # middle
    (17, 18),
    (18, 19),
    (19, 4),  # pinky
]

HAND_ID2NAME: dict[int, str] = {
    0: "THUMB_TIP",
    1: "INDEX_FINGER_TIP",
    2: "MIDDLE_FINGER_TIP",
    3: "RING_FINGER_TIP",
    4: "PINKY_TIP",
    5: "WRIST",
    6: "THUMB_CMC",
    7: "THUMB_MCP",
    8: "INDEX_FINGER_MCP",
    9: "INDEX_FINGER_PIP",
    10: "INDEX_FINGER_DIP",
    11: "MIDDLE_FINGER_MCP",
    12: "MIDDLE_FINGER_PIP",
    13: "MIDDLE_FINGER_DIP",
    14: "RING_FINGER_MCP",
    15: "RING_FINGER_PIP",
    16: "RING_FINGER_DIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PALM",
}

HAND_IDS: list[int] = [int(key) for key in HAND_ID2NAME]
