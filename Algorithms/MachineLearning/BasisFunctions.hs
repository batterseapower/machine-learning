-- | Basis functions of various kinds, useful for e.g. use with the LinearRegression module
module Algorithms.MachineLearning.BasisFunctions where

import Algorithms.MachineLearning.Utilities


-- | /Unnormalized/ 1D Gaussian, suitable for use as a basis function.
gaussianBasis :: Double -- ^ Mean of the Gaussian
              -> Double -- ^ Standard deviation of the Gaussian
              -> Double -- ^ Point on X axis to sample
              -> Double
gaussianBasis mean stdev x = exp (negate $ (square (x - mean)) / (2 * (square stdev)))