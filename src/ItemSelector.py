import cv2 as cv
from src.Utils import get_centroid

class ItemSelector:
    def __init__(self, work_size, min_area):
        self.__work_size = work_size
        self.__min_area = min_area

        self.__canny_threshold1 = 100
        self.__canny_threshold2 = 200

    def __get_contours(self, img):
        canny = cv.Canny(img, self.__canny_threshold1, self.__canny_threshold2)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        closing = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)

        contours, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return contours

    def __filter_contours(self, contours):
        area_filter = lambda contour: cv.contourArea(contour) >= self.__min_area
        return list(filter(area_filter, contours))

    # assumption that the contour of the polygon is the rightmost
    def __get_polygon_contour(self, contours):
        max_x = -1.0
        polygon_cnt = None

        for cnt in contours:
            centroid = get_centroid(cnt)
            if centroid[0] > max_x:
                max_x = centroid[0]
                polygon_cnt = cnt

        return polygon_cnt

    def select(self, img):
        resized = cv.resize(img, self.__work_size)
        contours = self.__get_contours(resized)
        contours = self.__filter_contours(contours)
        polygon_cnt = self.__get_polygon_contour(contours)
        obj_contours = [cnt for cnt in contours if cnt is not polygon_cnt]

        epsilon = 0.05 * cv.arcLength(polygon_cnt, True)
        polygon_cnt = cv.approxPolyDP(polygon_cnt, epsilon, True)
        return obj_contours, polygon_cnt
