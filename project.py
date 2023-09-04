import argparse
from PIL import Image
import numpy as np




def main():
    """
    check if input entered and/or exist
    """
    parser = argparse.ArgumentParser(description="taking input image")
    parser.add_argument("input", help="input image")
    args = parser.parse_args()
    try:
        img = Image.open(args.input)
    except IOError:
        print("image does not exist")
    Grayscale(img)
    grey = Grayscale(img)
    gaussian = Gaussian_filter(grey)
    gausimg, degree = find_edge(gaussian)
    non_max = non_max_supression(gausimg, degree)
    hyster, strong = double_thres(non_max)
    edge_img = edge_track(hyster, strong)
    edge_img = edge_img.resize((img.width, img.height))
    edge_img.save("edge_tracking.png")


# create grayscale image
def Grayscale(imgin):
    width, height = imgin.size
    shape = 0
    if height != width:
        if height > width:
            shape = height
            imgin = imgin.resize((shape, shape))
        elif width > height:
            shape = width
            imgin = imgin.resize((shape, shape))

    else:
        shape = width

    grey = np.empty((shape, shape))
    clr = np.array(imgin)
    for i in range(len(grey)):
        for j in range(len(grey[i])):
            blue, green, red = clr[i][j][0], clr[i][j][1], clr[i][j][2]
            grey[i][j] = int(round(0.299 * red + 0.587 * green + 0.114 * blue))

    grey = grey / np.amax(grey)
    grey = grey * 255
    grey = grey.astype(np.uint8)
    gray = Image.fromarray(grey)

    gray.save("grayscale.png")

    return grey


# create filtered image
def Gaussian_filter(grey):
    width, height = len(grey), len(grey[0])
    fltr = 5
    k = (fltr - 1) / 2
    sigma = 1.4
    front = 1 / (2 * np.pi * sigma**2)
    denom = 2 * np.square(sigma)
    gauss_filter = np.zeros((fltr, fltr))

    # kernel formed
    for i in range(fltr):
        for j in range(fltr):
            numerator = -1 * ((i + 1 - (k + 1)) ** 2 + (j + 1 - (k + 1)) ** 2)

            gauss_filter[i][j] = front * np.exp(numerator / denom)

    gauss = np.zeros_like(grey)
    pad = 2
    gry = np.pad(grey, pad, mode="constant", constant_values=0)

    # gaussian filter calc
    sqr = np.empty((fltr, fltr))
    for i in range(width):
        for j in range(height):
            sqr = gry[i : i + 5, j : j + 5]
            gauss[i][j] = round(np.sum(np.multiply(gauss_filter, sqr)))

    gauss = gauss.astype(np.uint8)
    gaussian = Image.fromarray(gauss)

    gaussian.save("GaussianBlur.png")
    return gaussian


# Find the intensity gradients of the image
def find_edge(image):
    w, h = image.size
    gx, gy = np.array(image), np.array(image)
    gx = np.pad(gx, pad_width=1, mode="constant", constant_values=0)
    gy = np.pad(gy, pad_width=1, mode="constant", constant_values=0)

    prewittx = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
    prewitty = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
    for i in range(w):
        for j in range(h):
            gx[i][j] = (prewittx * gx[i : i + 3, j : j + 3]).sum()
            gy[i][j] = (prewitty * gy[i : i + 3, j : j + 3]).sum()
    G = np.sqrt(np.add(gx**2, gy**2))
    G = (G / np.amax(G)) * 255

    theta = np.arctan2(gy, gx)

    return G, theta


# Apply gradient magnitude thresholding or lower bound cut-off suppression to get rid of spurious response to edge detection
def non_max_supression(inputimg, angle):
    angle = angle * (180 / np.pi)
    w, h = inputimg.shape
    non_max = np.empty((w, h))

    k = 255
    m = 255
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            if 0 <= angle[i][j] < 22.5 or 157.5 <= angle[i][j] <= 180:
                k = inputimg[i][j - 1]
                m = inputimg[i][j + 1]
            elif 22.5 <= angle[i][j] < 67.5:
                k = inputimg[i - 1][j - 1]
                m = inputimg[i + 1][j + 1]
            elif 67.5 <= angle[i][j] < 112.5:
                k = inputimg[i - 1][j - 1]
                m = inputimg[i + 1][j + 1]
            elif 112.5 <= angle[i][j] < 157.5:
                k = inputimg[i - 1][j]
                m = inputimg[i + 1][j]
            if inputimg[i][j] >= k and inputimg[i][j] >= m:
                non_max[i][j] = inputimg[i][j]
            else:
                non_max[i][j] = 0

    non_max = non_max.astype(np.uint8)
    supression = Image.fromarray(non_max)

    supression.save("NonMaxSupression.png")
    return supression


# Apply double threshold to determine potential edges
def double_thres(img, weak=0.9, strong=0.96):
    img = np.array(img)

    max = np.amax(img)
    high = round(max * strong)
    low = round(max * weak)

    lowi, lowj = np.where(img <= low)
    midi, midj = np.where((img > low) & (img < high))

    img[lowi][lowj] = 0
    img[midi][midj] = low

    img = img.astype(np.uint8)
    threshold = Image.fromarray(img)
    threshold.save("double_threshold.png")
    return threshold, high


# Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
def edge_track(input, edge):
    imageInfo = np.array(input)

    edges = np.zeros_like(imageInfo)
    for i in range(len(imageInfo)):
        for j in range(len(imageInfo[0])):
            compare = imageInfo[i : i + 3, j : j + 3]
            compare = compare.flatten()
            if any(num >= edge for num in compare):
                edges[i][j] = imageInfo[i][j]
    edges = edges.astype(np.uint8)
    Edge = Image.fromarray(edges)
    return Edge


if __name__ == "__main__":
    main()
