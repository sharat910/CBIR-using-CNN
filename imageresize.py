from PIL import Image

from resizeimage import resizeimage

image_name="ghoda.jpg"
new_name="new"+image_name
with open(image_name, 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [32, 32], validate=False)
        cover.save(new_name, image.format)
