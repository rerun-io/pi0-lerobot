from argparse import ArgumentParser

import gradio as gr
from pi0_lerobot.gradio_ui.pose_estimation_ui import pose_estimation_block

# from rerun_prompt_da.gradio_ui.prompt_da_ui import prompt_da_block

title = """# Multiview Exo Ego Pose Estimation for Robotics Motion Retargeting"""
description1 = """
    <a title="Website" href="https://promptda.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
    </a>
    <a title="arXiv" href="https://arxiv.org/abs/2403.20309" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
    </a>
    <a title="Github" href="https://github.com/rerun-io/prompt-da" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/github/stars/rerun-io/prompt-da?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
    </a>
    <a title="Social" href="https://x.com/pablovelagomez1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
    </a>
"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description1)
    with gr.Tab(label="Pose Estimation"):
        pose_estimation_block.render()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo.launch(share=args.share)
