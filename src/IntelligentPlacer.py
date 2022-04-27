import cv2 as cv

from src.ItemSelector import ItemSelector
from src.GeneticPacking import GeneticPacking

def check_image(path_to_image, work_info=False):
    img = cv.imread(path_to_image)
    selector = ItemSelector(work_size=(608, 608), min_area=800, work_info=work_info)
    obj_contours, poly_countour = selector.select(img)
    packing = GeneticPacking(poly_countour, work_info=work_info)
    return packing.pack(obj_contours)