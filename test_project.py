from CS50P.final_project.project import Grayscale, find_edge, double_thres, edge_track
import numpy as np
from PIL import Image


def test_Grayscale():
    input = Image.open("testimage.png")
    w, h = input.width, input.height
    print(w, h)
    if w != h:
        if w > h:
            size = len(w)
        else:
            size = h
    else:
        size = w

    img = input.resize((size, size))
    color = np.array(img)
    gray = np.array(img.convert(mode="L"))
    try:
        gray.shape == color.shape
    except Exception:
        assert True


def test_Gaussfilter():
    gray = Image.open("grayscale.png")
    Gauss = Image.open("GaussianBlur.png")

    if np.shape(np.array(gray)) != np.shape(np.array(Gauss)):
        pass


def test_edge():
    gauss = np.pad(
        np.array(Image.open("GaussianBlur.png")),
        pad_width=1,
        mode="constant",
        constant_values=0,
    )
    input = Image.open("GaussianBlur.png")
    direction, angle = find_edge(input)

    assert gauss.shape == angle.shape


def test_non_max():
    non_max = np.array(Image.open("NonMaxSupression.png"))
    gaublur = np.pad(
        np.array(Image.open("GaussianBlur.png")),
        pad_width=1,
        mode="constant",
        constant_values=0,
    )
    if non_max.shape != gaublur.shape:
        raise (ValueError)
    else:
        pass


def test_threshold():
    img, thres = double_thres(Image.open("NonMaxSupression.png"))
    assert img.size == Image.open("double_threshold.png").size


def test_edge_track():
    assert (
        edge_track(Image.open("double_threshold.png"), 0.9).size
        != Image.open("edge_tracking.png").size
    )
