{-# LANGUAGE PatternSignatures #-}

-- | Linear regression models, as discussed in chapter 3 of Bishop.
module Algorithms.MachineLearning.LinearRegression (
        LinearModel,
        regressLinearModel, regressRegularizedLinearModel, bayesianLinearRegression
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


data BayesianVarianceModel input = BayesianVarianceModel {
        bvm_basis_fns :: [input -> Target],
        bvm_weight_covariance :: Matrix Weight,
        bvm_beta :: Precision
    }

instance Show (BayesianVarianceModel input) where
    show model = "Weight Covariance: " ++ show (bvm_weight_covariance model) ++ "\n" ++
                 "Beta: " ++ show (bvm_beta model)

instance Model BayesianVarianceModel where
    predict model input = recip (bvm_beta model) + phi_app_x <.> (bvm_weight_covariance model <> phi_app_x)
      where
        phi_app_x = applyVector (bvm_basis_fns model) input


regressDesignMatrix :: (Vectorable input) => [input -> Target] -> Matrix Double -> Matrix Double
regressDesignMatrix basis_fns inputs
  = applyMatrix (map (. fromVector) basis_fns) inputs -- One row per sample, one column per basis function

-- | Regularized pseudo-inverse of a matrix, with regularization coefficient lambda.
regularizedPinv :: RegularizationCoefficient -> Matrix Double -> Matrix Double
regularizedPinv lambda phi = regularizedPrePinv lambda 1 phi <> trans phi

-- | Just the left portion of the formula for the pseudo-inverse, with coefficients alpha and beta, i.e.:
--
-- > (alpha * _I_ + beta * _phi_ ^ T * _phi_) ^ -1
regularizedPrePinv :: Precision -> Precision -> Matrix Double -> Matrix Double
regularizedPrePinv alpha beta phi = inv $ (alpha .* (ident (cols phi))) + (beta .* (trans phi <> phi))


-- | Regress a basic linear model with no regularization at all onto the given data using the
-- supplied basis functions.
--
-- The resulting model is likely to suffer from overfitting, and may not be well defined in the basis
-- functions are close to colinear.
--
-- However, the model will be the optimal model for the data given the basis in least-squares terms.  It
-- is also very quick to find, since there is a closed form solution.
--
-- Equation 3.15 in Bishop.
regressLinearModel :: (Vectorable input) => [input -> Target] -> DataSet -> LinearModel input
regressLinearModel basis_fns ds = LinearModel { lm_basis_fns = basis_fns, lm_weights = weights }
  where
    design_matrix = regressDesignMatrix basis_fns (ds_inputs ds)
    weights = pinv design_matrix <> ds_targets ds

-- | Regress a basic linear model with a sum-of-squares regularization term.  This penalizes models with weight
-- vectors of large magnitudes and hence ameliorates the over-fitting problem of 'regressLinearModel'.
-- The strength of the regularization is controlled by the lambda parameter.  If lambda is 0 then this function
-- is equivalent to the unregularized regression.
--
-- The resulting model will be optimal in terms of least-squares penalized by lambda times the sum-of-squares of
-- the weight vector.  Like 'regressLinearModel', a closed form solution is used to find the model quickly.
--
-- Equation 3.28 in Bishop.
regressRegularizedLinearModel :: (Vectorable input) => RegularizationCoefficient -> [input -> Target] -> DataSet -> LinearModel input
regressRegularizedLinearModel lambda basis_fns ds = LinearModel { lm_basis_fns = basis_fns, lm_weights = weights }
  where
    design_matrix = regressDesignMatrix basis_fns (ds_inputs ds)
    weights = regularizedPinv lambda design_matrix <> ds_targets ds

-- | Bayesian linear regression, using an isotropic Gaussian prior for the weights centred at the origin.  The precision
-- of the weight prior is controlled by the parameter alpha, and our belief about the inherent noise in the data is
-- controlled by the precision parameter beta.
--
-- Bayesion linear regression with this prior is entirely equivalent to calling 'regressRegularizedLinearModel' with
-- lambda = alpha / beta.  However, the twist is that we can use our knowledge of the prior to also make an estimate
-- for the variance of the true value about any input point.
--
-- Equations 3.53, 3.54 and 3.59 in Bishop.
bayesianLinearRegression :: (Vectorable input) 
                         => Precision -- ^ Precision of Gaussian weight prior
                         -> Precision -- ^ Precision of noise on samples
                         -> [input -> Target] -> DataSet -> (LinearModel input, BayesianVarianceModel input)
bayesianLinearRegression alpha beta basis_fns ds
  = (LinearModel { lm_basis_fns = basis_fns, lm_weights = weights },
     BayesianVarianceModel { bvm_basis_fns = basis_fns, bvm_weight_covariance = weight_covariance, bvm_beta = beta })
  where
    design_matrix = regressDesignMatrix basis_fns (ds_inputs ds)
    weight_covariance = regularizedPrePinv alpha beta design_matrix
    weights = beta .* weight_covariance <> trans design_matrix <> (ds_targets ds)