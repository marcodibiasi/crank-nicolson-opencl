import numpy as np
from PIL import Image
from noise import pnoise2

def save_pgm(filename, data):
    height, width = data.shape

    with open(filename, 'w') as f:
        f.write(f"{width} {height}\n")
        for row in data:
            line = ' '.join(str(val) for val in row)
            f.write(line + "\n")

def save_png(filename, data):
    img = Image.fromarray(data)
    img.save(filename)

def generate_simple_perlin(width, height):
    scale = width/6
    data = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            val = pnoise2(x / scale, y / scale, octaves=1)
            norm = (val + 1) / 2  
            contrast = norm**2
            data[y, x] = int(contrast * 255)
    return data

if __name__ == "__main__":
    width, height = 1024, 1024
    heatmap = generate_simple_perlin(width, height)
    save_pgm("heatmap.pgm", heatmap)
    print("Heatmap saved in heatmap.pgm")

    save_png("heatmap.png", heatmap)
    print("Heatmap saved in heatmap.png\n")