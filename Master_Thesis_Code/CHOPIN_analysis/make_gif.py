import glob
from pathlib import Path

import numpy as np
import tyro
from PIL import Image


def get_image_number(png_path: str) -> int:
    return int(png_path.split("_")[-1].split(".")[0])

def main(gif_folder:Path, save_path:Path):
     
    glob_string = str(gif_folder.absolute()) +"/*.png"
    files = glob.glob(glob_string)
    images = {}
    for file in files:
        image_number = get_image_number(file)
        images[image_number] = Image.open(file)

    image_array = []
    min_image = np.min(list(images.keys()))
    max_image = np.max(list(images.keys()))
    for i in range(min_image, max_image):
        try:
            image_array.append(images[i])
        except:
            continue

    image_array[0].save('globe.gif', format='GIF', 
               append_images=image_array[0:], save_all=True, duration=200, 
               loop=0, optimize=False)

tyro.cli(main)