### PIL Crop
# Works perfect!

from PIL import Image
import os

def crop(path, input):
    im = Image.open(os.path.join(path, input))
    img_width, img_height = im.size
    height = img_height // 5
    width = img_width // 10
    for i in range(0, img_height, height):
        for j in range(0, img_width, width):
            box = (j, i, j+width, i+height)
            o = im.crop(box)
            output = str(i) + str(j) + input
            o.save(os.path.join(path, "%s" % output), "PNG")


crop("C:/Users/brand/Downloads/", "teaser_hejkLDN.jpg")


### PIL Subscript
# Poorly documented, non-working.

def tile_ps():
    # M = 5  # 1020 / 200 px
    # N = 10  # 3200 / 320 px
    from PIL import Image

    with Image.open("C:/Users/brand/Downloads/teaser_hejkLDN.jpg") as im:
        width, height = im.size
        M = width // 5
        N = height // 10
        px = im.load()

        tiles = [px[x:x + M, y:y + N] for x in range(0, width, M) for y in range(0, height, N)]

# tile_ps()


### itertools
# Great for square tiles, not so much for rectangular.

import os

from PIL import Image
from itertools import product


def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
    for i, j in grid:
        box = (j, i, j + d, i + d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)


# tile('teaser_hejkLDN.jpg', 'C:/Users/brand/Downloads', 'C:/Users/brand/PycharmProjects/cae/images/skele', 50)
