import streamlit as st
import torch
import re
import numpy as np
import imageio
from diffusers import StableDiffusionPipeline
from sgm.modules.diffusionmodules.sampling import EulerAncestralSampler

# Define the model and device
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pipeline
PIPE = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
PIPE = PIPE.to(DEVICE)

# Define the versions and their specifications
VERSION2SPECS = {
    "SDXL-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_turbo_1.0.safetensors",
    },
    "SD-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/sd_turbo.safetensors",
    },
}

class SubstepSampler(EulerAncestralSampler):
    def __init__(self, n_sample_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sample_steps = n_sample_steps
        self.steps_subset = [0, 100, 200, 300, 1000]

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        sigmas = sigmas[
            self.steps_subset[: self.n_sample_steps] + self.steps_subset[-1:]
        ]
        uc = cond
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc

def generate_images(prompt):
    subjects = re.findall(r"\{([^}]+)\}", prompt)
    if subjects:
        images = []
        for subject in subjects:
            modified_prompt = prompt.replace("{" + subject + "}", subject)
            image = PIPE(modified_prompt).images[0]
            images.append(image)
        return images
    else:
        image = PIPE(prompt).images[0]
        return [image]

def generate_video(image):
    frames = [np.array(image)] * 30  # Repeat image to create a short video
    video = imageio.get_writer("output_video.mp4", fps=10)  # Create video writer
    for frame in frames:
        video.append_data(frame)
    video.close()

def display_images(images):
    for i, image in enumerate(images):
        st.image(image, caption=f"Image {i+1}", use_column_width=True)
        st.download_button(
            label="Download Image",
            data=image.tobytes(),
            file_name=f"image_{i+1}.png",
            mime="image/png",
        )

def display_video():
    st.video("output_video.mp4")  # Display video
    st.markdown("[Download Video](output_video.mp4)")

def main():
    st.title("Text-to-Image Generator")

    # Text input for prompt
    prompt = st.text_input("Enter your prompt:")

    # Button to generate images
    if st.button("Generate Images"):
        with st.spinner("Generating..."):
            images = generate_images(prompt)
        
        # Display radio buttons to select image version
        selected_version = st.radio("Select Image Version:", list(VERSION2SPECS.keys()))

        # Display images with download buttons
        display_images(images)

        # Button to generate video from selected image
        if st.button("Generate Video"):
            selected_image = images[0]  # For simplicity, select the first image
            generate_video(selected_image)
            display_video()

if __name__ == "__main__":
    main()
