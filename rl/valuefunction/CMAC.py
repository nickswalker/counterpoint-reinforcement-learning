#!/usr/bin/python
"""
Author: Jeremy M. Stober
Program: TILES.PY
Date: Monday, March 31 2008
Description: A simple CMAC implementation.
"""

import pickle

from numpy import *


class CMAC(object):
    def __init__(self, nlevels, quantization, beta):
        self.nlevels = nlevels
        self.quantization = quantization
        self.weights = {}
        self.beta = beta

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'), pickle.HIGHEST_PROTOCAL)

    def quantize(self, vector):
        """
		Generate receptive field coordinates for each level of the CMAC.
		"""

        quantized = (vector / self.quantization).astype(int)
        coords = []

        for i in range(self.nlevels):
            # Note that the tile size is nlevels * quantization!

            # Coordinates for this tile.
            point = list(quantized - (quantized - i) % self.nlevels)

            # Label the ith tile so that it gets hashed uniquely.
            point.append(i)

            coords.append(tuple(point))

        return coords

    def difference(self, vector, delta, quantized=False):
        """
		Train the CMAC using the difference instead of the response.
		"""

        # Coordinates for each level tiling.
        coords = None
        if quantized == False:
            coords = self.quantize(vector)
        else:
            coords = vector

        error = self.beta * delta  # delta = response - prediction

        for pt in coords:
            self.weights[pt] += error

        return delta

    def train(self, vector, response, quantized=False):
        """
		Train the CMAC.
		"""

        # Coordinates for each level tiling.
        coords = None
        if quantized == False:
            coords = self.quantize(vector)
        else:
            coords = vector

        # Use Python's own hashing for storing feature weights. If you
        # roll your own you'll have to learn about Universal Hashing.
        prediction = sum([self.weights.setdefault(pt, 0.0) for pt in coords]) / len(coords)
        error = self.beta * (response - prediction)

        for pt in coords:
            self.weights[pt] += error

        return prediction

    def __len__(self):
        return len(self.weights)

    def eval(self, vector, quantized=False):
        """
		Eval the CMAC.
		"""

        # Coordinates for each level tiling.
        coords = None
        if quantized == False:
            coords = self.quantize(vector)
        else:
            coords = vector

        return sum([self.weights.setdefault(pt, 0.0) for pt in coords]) / len(coords)
