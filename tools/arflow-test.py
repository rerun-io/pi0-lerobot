#!/usr/bin/env python3
"""Demonstrates the most barebone usage of ARFlow."""

from __future__ import annotations

import sys

import arflow


class MinimalService(arflow.ARFlowService):
    def on_register(self, request: arflow.RegisterRequest):
        pass

    def on_frame_received(self, frame_data: arflow.DataFrameRequest):
        pass


def main() -> None:
    # sanity-check since all other example scripts take arguments:
    assert len(sys.argv) == 1, f"{sys.argv[0]} does not take any arguments"
    arflow.create_server(MinimalService, port=8500, path_to_save=None)


if __name__ == "__main__":
    main()
