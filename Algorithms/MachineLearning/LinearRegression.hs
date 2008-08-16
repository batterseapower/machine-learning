{-# LANGUAGE PatternSignatures #-}

-- | Linear regression models, as discussed in chapter 3 of Bishop.
module Algorithms.MachineLearning.LinearRegression (
        LinearModel, regressLinearModel, regressRegularizedLinearModel
    ) where

import Algorithms.MachineLearning.Framework
import Algorithms.MachineLearning.LinearAlgebra


data LinearModel input = LinearModel {
        lm_basis_fns :: [input -> Target],
        lm_weights   :: Vector Weight
    }

instance Show (LinearModel input) where
    show model = "Weights: " ++ show (lm_weights model)

instance Model LinearModel where
    predict model input = (lm_weights model) <.> phi_app_x
      where
        phi_app_x = applyVector (lm_basis_fns model) input


-- | Regress a basic linear model with no regularization at all onto the given data using the
-- supplied basis functions.
--
-- The resulting model is likely to suffer from overfitting, and may not be well defined in the basis
-- functions are close to colinear.
--
-- However, the model will be the optimal model for the data given the basis in least-squares terms.  It
-- is also very quick to find, since there is a closed form solution.
regressLinearModel :: (Vectorable input) => [input -> Target] -> DataSet input -> LinearModel input
regressLinearModel = regressLinearModelCore pinv

-- | Regress a basic linear model with a sum-of-squares regularization term.  This penalizes models with weight
-- vectors of large magnitudes and hence ameliorates the over-fitting problem of 'regressLinearModel'.
-- The strength of the regularization is controlled by the lambda parameter.
--
-- The resulting model will be optimal in terms of least-squares penalized by lambda times the sum-of-squares of
-- the weight vector.  Like 'regressLinearModel', a closed form solution is used to find the model quickly.
regressRegularizedLinearModel :: (Vectorable input) => RegularizationCoefficient -> [input -> Target] -> DataSet input -> LinearModel input
regressRegularizedLinearModel lambda = regressLinearModelCore regularizedPinv
  where
    regularizedPinv phi = let trans_phi = trans phi
                          in inv ((lambda .* (ident (cols phi))) + (trans_phi <> phi)) <> trans_phi

regressLinearModelCore :: (Vectorable input) => (Matrix Double -> Matrix Double) -> [input -> Target] -> DataSet input -> LinearModel input
regressLinearModelCore find_pinv basis_fns ds
  = LinearModel { lm_basis_fns = basis_fns, lm_weights = weights }
  where
    designMatrix = applyMatrix (map (. fromVector) basis_fns) (ds_inputs ds) -- One row per sample, one column per basis function
    weights = find_pinv designMatrix <> (ds_targets ds)