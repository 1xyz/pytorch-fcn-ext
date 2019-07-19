from enum import Enum
from typing import Tuple

from PIL.Image import Image, FLIP_TOP_BOTTOM, FLIP_LEFT_RIGHT
from torchvision.transforms import Resize, CenterCrop


class ImageTransformType(Enum):
    Unknown = 0
    TopRightCrop = 1
    BottomRightCrop = 2
    TopLeftCrop = 3
    BottomLeftCrop = 4
    CenterCrop = 5
    Resize = 6


class FlipType(Enum):
    Unknown = 0
    Horizontal = 1
    Vertical = 2


def apply_transform(img: Image, lbl: Image,
                    img_transform_type: ImageTransformType,
                    flip_type: FlipType,
                    th: int, tw: int) -> Tuple[Image, Image]:
    if img_transform_type not in TransformDict:
        return img, lbl
    f = TransformDict[img_transform_type]
    next_img, next_lbl = img, lbl
    if flip_type == FlipType.Horizontal:
        next_img, next_lbl = hflip(img), hflip(lbl)
    elif flip_type == FlipType.Vertical:
        next_img, next_lbl = vflip(img), vflip(lbl)
    return f(next_img, th, tw), f(next_lbl, th, tw)


def top_left_crop(img: Image, crop_h: int, crop_w: int) -> Image:
    return img.crop((0, 0, crop_w, crop_h))


def top_right_crop(img: Image, crop_h: int, crop_w: int) -> Image:
    w, h = img.size
    return img.crop((w - crop_w, 0, w, crop_h))


def bottom_left_crop(img: Image, crop_h: int, crop_w: int) -> Image:
    w, h = img.size
    return img.crop((0, h - crop_h, crop_w, h))


def bottom_right_crop(img: Image, crop_h: int, crop_w: int) -> Image:
    w, h = img.size
    return img.crop((w - crop_w, h - crop_h, w, h))


def center_crop(img: Image, crop_h: int, crop_w: int) -> Image:
    return CenterCrop((crop_h, crop_w))(img)


def resize(img: Image, new_h: int, new_w: int) -> Image:
    return Resize([new_h, new_w])(img)


def vflip(img: Image) -> Image:
    return img.transpose(FLIP_TOP_BOTTOM)


def hflip(img: Image) -> Image:
    return img.transpose(FLIP_LEFT_RIGHT)


TransformDict = {
    ImageTransformType.TopLeftCrop: top_left_crop,
    ImageTransformType.TopRightCrop: top_right_crop,
    ImageTransformType.Resize: resize,
    ImageTransformType.BottomLeftCrop: bottom_left_crop,
    ImageTransformType.BottomRightCrop: bottom_right_crop,
    ImageTransformType.CenterCrop: center_crop,
}
