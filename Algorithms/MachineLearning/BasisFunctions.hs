-- | Basis functions of various kinds, useful for e.g. use with the LinearRegression module
module Algorithms.MachineLearning.BasisFunctions where

import Algorithms.MachineLearning.Framework
import Algorithms.MachineLearning.LinearAlgebra
import Algorithms.MachineLearning.Utilities


-- | Basis function that is 1 everywhere
constantBasis :: a -> Double
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

-- | /Unnormalized/ multi-dimensional Gaussian, suitable for use as a basis function.
multivariateGaussianBasis :: Vector Mean     -- ^ Mean of the Gaussian
                          -> Matrix Variance -- ^ Covariance matrix
                          -> Vector Double   -- ^ Point to sample
                          -> Double
multivariateGaussianBasis mean covariance x = exp (negate $ (deviation <.> (inv covariance <> deviation)) / 2)
  where deviation = x - mean

-- | Family of multi-dimensional gaussian basis functions with constant, isotropic variance and
-- the given means, with a constant basis function to capture the mean of the target variable.
multivariateIsotropicGaussianBasisFamily :: [Vector Mean] -> Variance -> [Vector Double -> Double]
multivariateIsotropicGaussianBasisFamily means common_variance = constantBasis : map (flip multivariateGaussianBasis covariance) means
  where covariance = (1 / common_variance) .* ident (dim (head means))