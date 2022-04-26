import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt

from numpy.random import dirichlet, uniform
from numpy.random import choice, randint

from src.Utils import bar2cart, random_in_range
from src.Utils import warp_contour, normalize_contour

def tournament(chromosomes, scores):
    # Deterministic tournament selection
    return chromosomes[np.argmax(scores)]

def outbreeding(selection):
    first_idx = randint(len(selection))
    first_parent = selection[first_idx]

    max_dist = -1.0
    second_parent = None
    for chromosome in selection:
        if chromosome is first_parent:
            continue

        diff = chromosome - first_parent
        dist = np.linalg.norm(diff)
        if dist > max_dist:
            second_parent = chromosome
            max_dist = dist

    return first_parent, second_parent

#The packing position search algorithm is based on the genetic algorithm
class GeneticPacking:
    #  bound_contour - bounding polygon vertex sequence (requires convexity and connectivity)
    def __init__(self, bound_contour):
        reshaped = bound_contour.reshape(-1, 2)
        # Introduced heuristic - circumscribe a bounding box (below, BOX) around the original polygon
        x, y, w, h = cv.boundingRect(reshaped)
        reshaped -= np.array([x, y])

        # mask to fill the BOX with objects
        self.__fill_mask = np.zeros([h, w], dtype=np.uint8)
        # mask to fill the BOX with polygon
        self.__cut_mask = np.zeros([h, w], dtype=np.uint8)
        cv.fillPoly(self.__cut_mask, [bound_contour], 255)

        self.__bound_cnt = reshaped
        self.__dim = bound_contour.shape[0]
        self.__alpha = 1.5 * np.ones(self.__dim)

        self.__contours_n = 0
        self.__contours = None
        self.__contours_area = 0

        self.__population_size = 100
        self.__population = None
        self.__scores = np.zeros(self.__population_size)

        self.__children_fraction = 0.85
        self.__crossover_fraction = 0.5

        self.__mutate_eta = 0.7
        self.__mutate_mu = 0.3

        self.__tournament_iter = 60
        self.__tournament_size = 4

        self.__max_iter = 100

        self.__rng = np.random.default_rng()

    # packing function
    # contours - description array of objects (sequence of vertices)
    def pack(self, contours, epsilon=10**-1):
        # calculation of the total area of objects
        self.__set_contours(contours)

        # genetic algorithm regarding the position parameters of objects in a BOX
        self.__init_population()
        for i in range(self.__max_iter):
            self.__update_scores()
            max_score = np.max(self.__scores)

            if max_score > 1.0 - epsilon:
                return True
            self.__generate_population()
        return False

    def __set_contours(self, contours):
        self.__contours_area = 0
        self.__contours = []

        for cnt in contours:
            # centering about the center of mass
            normalized = normalize_contour(cnt)
            self.__contours_area += cv.contourArea(normalized)

            reshaped = normalized.reshape(-1, 2)
            self.__contours.append(reshaped)

        self.__contours_n = len(self.__contours)

    def __init_population(self):
        self.__population = np.array([self.__create_chromosome()
                                      for _ in range(self.__population_size)])

    # formation of a genetic algorithm unit - descriptions of an array of objects (below, CHROMOSOME)
    def __create_chromosome(self):
        size = self.__contours_n
        chromosome = np.zeros((size, self.__dim + 1))
        # coordinates of the center of mass of objects in the barycentric basis of the bounding contour
        chromosome[:, :-1] = dirichlet(self.__alpha, size)
        # object rotation angle
        chromosome[:, -1] = uniform(0, 1.0, size)
        # note that at this stage we do not care about the intersections of objects
        return chromosome

    def __update_scores(self):
        self.__scores = np.array([self.__obj_function(chromosome)
                                  for chromosome in self.__population])

    # score function for CHROMOSOME
    def __obj_function(self, chromosome):
        # represent objects in the Cartesian basis
        # with respect to the barocentric representation in the CHROMOSOME
        placements = [self.__apply_placement(cnt, params)
                     for cnt, params in zip(self.__contours, chromosome)]

        # fill the BOX with objects
        self.__fill_mask.fill(0)
        for cnt in placements:
            cv.fillPoly(self.__fill_mask, [cnt], 255)

        # intersect the areas of the original contour with areas of figures
        cut = cv.bitwise_and(self.__fill_mask, self.__cut_mask)

        # the key criterion for the quality of the resulting packaging will be
        # the ratio of the total area of objects to the area of packed objects
        # (tending to zero in the correct case).
        # note that in this case, all possible intersections of objects,
        # as well as objects falling out of the box, reduce the score.
        return np.count_nonzero(cut) / self.__contours_area

    def __apply_placement(self, cnt, params):
        alphas = params[:-1].reshape(-1, 1)
        centroid = bar2cart(self.__bound_cnt, alphas)

        phi = params[-1]
        angle = 2 * np.pi * phi
        return warp_contour(cnt, centroid, angle)

    # in what follows, we consider the algorithm for generating a new population,
    # which mostly consists of heuristics
    def __generate_population(self):
        selection = self.__select()

        children_n = self.__children_fraction * self.__population_size
        crossover_iter = int(children_n // 2)

        # part of the new population of CHROMOSOMES is obtained from
        # the usual crossing of the best CHROMOSOMES of the parental population
        children = []
        for _ in range(crossover_iter):
            child1, child2 = self.__crossover(*outbreeding(selection))
            children += [child1, child2]

        # the other part is derived by mutation
        mutant_n = self.__population_size - len(children)
        mutants = [self.__mutate(selection[i])
                    for i in randint(len(selection), size=mutant_n)]

        self.__population = np.array(children + mutants)

    def __select(self):
        selection = []
        # Small samples are selected from the general population of CHROMOSOMES
        # The best in the sample continue to participate in the selection
        for i in range(self.__tournament_iter):
            participants = choice(self.__population_size, self.__tournament_size)
            winner = tournament(self.__population[participants], self.__scores[participants])
            selection.append(winner)

        return np.array(selection)

    def __crossover(self, parent1, parent2):
        child1, child2 = parent1.copy(), parent2.copy()

        swap_size = int(self.__crossover_fraction * self.__contours_n)
        swap_seq = choice(self.__contours_n, swap_size)

        # Crossover means crossover of arrangements of some objects
        child1[swap_seq] = parent2[swap_seq]
        child2[swap_seq] = parent1[swap_seq]

        return child1, child2

    def __mutate(self, chromosome):
        # Mutation means mutation of some rotation angle
        alpha_mat = chromosome[:, :-1]
        sums = np.sum(alpha_mat, axis=0)

        min_alpha = np.argmin(sums)
        max_alpha = np.argmax(sums)

        mutant = chromosome.copy()

        diff = mutant[:, max_alpha] - mutant[:, min_alpha]
        alpha_bias = self.__mutate_eta * diff

        mutant[:, min_alpha] += alpha_bias
        mutant[:, max_alpha] -= alpha_bias

        angles = mutant[:, -1].copy()
        angle_bias = self.__mutate_mu

        angles += random_in_range(-angle_bias, angle_bias, angles.shape)
        angles = np.modf(angles)[0]
        angles[angles < 0] += 1

        mutant[:, -1] = angles
        return mutant
