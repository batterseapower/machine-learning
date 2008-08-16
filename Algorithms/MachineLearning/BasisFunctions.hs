module Algorithms.MachineLearning.BasisFunctions where

import Algorithms.MachineLearning.Utilities


normalGaussianBasis :: Double -> Double -> Double -> Double
normalGaussianBasis mean scale x = exp (negate $ (square (x - mean)) / (2 * (square scale)))