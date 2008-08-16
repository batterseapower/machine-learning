-- | Basis functions of various kinds, useful for e.g. use with the LinearRegression module
module Algorithms.MachineLearning.BasisFunctions where

import Algorithms.MachineLearning.Framework
import Algorithms.MachineLearning.Utilities


-- | /Unnormalized/ 1D Gaussian, suitable for use as a basis function.
gaussianBasis :: Mean     -- ^ Mean of the Gaussian
              -> Variance -- ^ Variance of the Gaussian
              -> Double   -- ^ Point on X axis to sample
              -> Double
gaussianBasis mean variance x = exp (negate $ (square (x - mean)) / (2 * variance))