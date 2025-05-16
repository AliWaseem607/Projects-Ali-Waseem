# %%
from pathlib import Path

import numpy as np
from PIL import Image

# %%
# Stitch rain period
image_1 = Image.open("data/quicklooks/0403.z1.png")
image_2 = Image.open("data/quicklooks/0503.z1.png")
image_3 = Image.open("data/quicklooks/0603.z1.png")

box_1 = (0, 0, 914, image_1.size[1])
box_2 = (104, 0, 914, image_2.size[1])
box_3 = (104, 0, image_3.size[0], image_3.size[1])

cropped_image_1 = image_1.crop(box_1)
cropped_image_2 = image_2.crop(box_2)
cropped_image_3 = image_3.crop(box_3)

new_width = cropped_image_1.size[0] + cropped_image_2.size[0] + cropped_image_3.size[0]
new_height = np.max((cropped_image_1.size[1], cropped_image_2.size[1], cropped_image_3.size[1]))

stitched_image = Image.new("RGBA", (new_width, new_height))
stitched_image.paste(cropped_image_1, box=(0, 0))
stitched_image.paste(cropped_image_2, box=(cropped_image_1.size[0], 0))
stitched_image.paste(cropped_image_3, box=(cropped_image_1.size[0] + cropped_image_2.size[0], 0))

# %%
# stitch evelopement
day_1_path = Path("data/quicklooks/1019/")
day_2_path = Path("data/quicklooks/1020")

pics = {}
for file in day_1_path.iterdir():
    key = day_1_path.stem + file.stem[:2]
    pics[key] = Image.open(file)

for file in day_2_path.iterdir():
    key = day_2_path.stem + file.stem[:2]
    pics[key] = Image.open(file)

order = np.linspace(101901, 101923, 23)
order = np.hstack([order, np.linspace(102000, 102017, 18)])
str_order = [str(int(x)) for x in order]

cropped_pics = {}
box_start = (0, 0, 912, 800)
box_middle = (106, 0, 912, 800)
box_end = (106, 0, 1200, 800)

for i in range(len(str_order)):
    key = str_order[i]
    if i == 0:
        cropped_pics[key] = pics[key].crop(box=box_start)
        continue
    if i == len(str_order) - 1:
        cropped_pics[key] = pics[key].crop(box=box_end)
        continue

    cropped_pics[key] = pics[key].crop(box=box_middle)

total_width = 0
for item in cropped_pics.values():
    total_width += item.size[0]

stitched_image = Image.new("RGBA", (total_width, 800))

running_width = 0
for i in range(len(str_order)):
    key = str_order[i]
    stitched_image.paste(cropped_pics[key], box=(running_width, 0))
    running_width += cropped_pics[key].size[0]


resized_image = stitched_image.resize((1200, 800))
