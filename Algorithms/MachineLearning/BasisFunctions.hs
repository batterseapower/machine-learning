-- | Basis functions of various kinds, useful for e.g. use with the LinearRegression module
module Algorithms.MachineLearning.BasisFunctions where

import Algorithms.MachineLearning.Framework
import Algorithms.MachineLearning.Utilities


-- | Basis function that is 1 everywhere
constantBasis :: Double -> Double
constantBasis = const 1

-- | /Unnormalized/ 1D Gaussian, suitable for use as a basis function.
gaussianBasis :: Mean     -- ^ Mean of the Gaussian
              -> Variance -- ^ Variance of the Gaussian
              -> Double   -- ^ Point on X axis to sample
              -> Double
gaussianBasis mean variance x = exp (negate $ (square (x - mean)) / (2 * variance))

-- | Family of gaussian basis functions with constant variance and the given means, with
-- a constant basis function to capture the mean of the target variable.
gaussianBasisFamily :: [Mean] -> Variance -> [Double -> Double]
gaussianBasisFamily means variance = constantBasis : map (flip gaussianBasis variance) means