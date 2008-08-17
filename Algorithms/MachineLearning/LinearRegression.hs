{-# LANGUAGE PatternSignatures #-}

-- | Linear regression models, as discussed in chapter 3 of Bishop.
module Algorithms.MachineLearning.LinearRegression (
        LinearModel,
        regressLinearModel, regressRegularizedLinearModel, regressBayesianLinearModel,
        regressEMBayesianLinearModel, regressFullyDeterminedEMBayesianLinearModel
    ) where

import Algorithms.MachineLearning.Framework
import Algorithms.MachineLearning.LinearAlgebra
import Algorithms.MachineLearning.Utilities


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
        bvm_inv_hessian :: Matrix Weight, -- Equivalent to the weight distribution covariance matrix
        bvm_beta :: Precision
    }

instance Show (BayesianVarianceModel input) where
    show model = "Inverse Hessian: " ++ show (bvm_inv_hessian model) ++ "\n" ++
                 "Beta: " ++ show (bvm_beta model)

instance Model BayesianVarianceModel where
    predict model input = recip (bvm_beta model) + (phi_app_x <> bvm_inv_hessian model) <.> phi_app_x --phi_app_x <.> (bvm_inv_hessian model <> phi_app_x)
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
regressLinearModel
    :: (Vectorable input) => [input -> Target] -> DataSet -> LinearModel input
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
regressRegularizedLinearModel
    :: (Vectorable input) => RegularizationCoefficient -> [input -> Target] -> DataSet -> LinearModel input
regressRegularizedLinearModel lambda basis_fns ds = LinearModel { lm_basis_fns = basis_fns, lm_weights = weights }
  where
    design_matrix = regressDesignMatrix basis_fns (ds_inputs ds)
    weights = regularizedPinv lambda design_matrix <> ds_targets ds


-- | Determine the mean weight and inverse hessian matrix given alpha, beta, the design matrix and the targets.
bayesianPosteriorParameters :: Precision -> Precision -> Matrix Double -> Vector Double -> (Vector Double, Matrix Double)
bayesianPosteriorParameters alpha beta design_matrix targets = (weights, inv_hessian)
  where
    inv_hessian = regularizedPrePinv alpha beta design_matrix
    weights = beta .* inv_hessian <> trans design_matrix <> targets

-- | Bayesian linear regression, using an isotropic Gaussian prior for the weights centred at the origin.  The precision
-- of the weight prior is controlled by the parameter alpha, and our belief about the inherent noise in the data is
-- controlled by the precision parameter beta.
--
-- Bayesion linear regression with this prior is entirely equivalent to calling 'regressRegularizedLinearModel' with
-- lambda = alpha / beta.  However, the twist is that we can use our knowledge of the prior to also make an estimate
-- for the variance of the true value about any input point.
--
-- Equations 3.53, 3.54 and 3.59 in Bishop.
regressBayesianLinearModel
    :: (Vectorable input)
    => Precision -- ^ Precision of Gaussian weight prior
    -> Precision -- ^ Precision of noise on samples
    -> [input -> Target] -> DataSet -> (LinearModel input, BayesianVarianceModel input)
regressBayesianLinearModel alpha beta basis_fns ds
  = (LinearModel { lm_basis_fns = basis_fns, lm_weights = weights },
     BayesianVarianceModel { bvm_basis_fns = basis_fns, bvm_inv_hessian = inv_hessian, bvm_beta = beta })
  where
    design_matrix = regressDesignMatrix basis_fns (ds_inputs ds)
    (weights, inv_hessian) = bayesianPosteriorParameters alpha beta design_matrix (ds_targets ds)

-- | Evidence-maximising Bayesian linear regression, using an isotropic Gaussian prior for the weights centred at the
-- origin.  The precision of the weight prior is controlled by the parameter alpha, and our belief about the inherent
-- noise in the data is controlled by the precision parameter beta.
--
-- This is similar to 'bayesianLinearRegression', but rather than just relying on the supplied values for alpha and beta
-- an iterative procedure is used to try and find values that are best supported by the supplied training data.  This is
-- an excellent way of finding a reasonable trade-off between over-fitting of the training set with a complex model and
-- accuracy of the model.
--
-- As a bonus, this function returns gamma, the effective number of parameters used by the regressed model.
--
-- Equations 3.87, 3.92 and 3.95 in Bishop.
regressEMBayesianLinearModel
    :: (Vectorable input)
    => Precision -- ^ Initial estimate of Gaussian weight prior
    -> Precision -- ^ Initial estimate for precision of noise on samples
    -> [input -> Target] -> DataSet -> (LinearModel input, BayesianVarianceModel input, EffectiveNumberOfParameters)
regressEMBayesianLinearModel initial_alpha initial_beta basis_fns ds
  = convergeOnEMBayesianLinearModel loopWorker design_matrix initial_alpha initial_beta basis_fns ds
  where
    n = fromIntegral $ dataSetSize ds
    
    design_matrix = regressDesignMatrix basis_fns (ds_inputs ds)
    -- The unscaled eigenvalues will be positive because phi ^ T * phi is positive definite.
    (unscaled_eigenvalues, _) = eigSH (trans design_matrix <> design_matrix)
    
    loopWorker alpha beta = (n - gamma, gamma)
      where
        -- We save computation by calculating eigenvalues once for the design matrix and rescaling each iteration
        eigenvalues = beta .* unscaled_eigenvalues
        gamma = vectorSum (eigenvalues / (addConstant alpha eigenvalues))

-- | Evidence-maximising Bayesian linear regression, using an isotropic Gaussian prior for the weights centred at the
-- origin.  The precision of the weight prior is controlled by the parameter alpha, and our belief about the inherent
-- noise in the data iscontrolled by the precision parameter beta.
--
-- This is similar to 'regressEMBayesianLinearModel', but suitable only for the situation where there is much more
-- training data than there are basis functions you want to assign weights to.  Due to the introduction of this
-- constraint, it is much faster than the other function and yet produces results of similar quality.
--
-- Like with 'regressEMBayesianLinearModel', the effective number of parameters, gamma, used by the regressed model
-- is returned. However, because for this function to make sense you need to be sure that there is sufficient data
-- that all the parameters are determined, the returned gamma is always just the number of basis functions (and
-- hence weights).
--
-- Equations 3.98 and 3.99 in Bishop.
regressFullyDeterminedEMBayesianLinearModel
    :: (Vectorable input)
    => Precision -- ^ Initial estimate of Gaussian weight prior
    -> Precision -- ^ Initial estimate for precision of noise on samples
    -> [input -> Target] -> DataSet -> (LinearModel input, BayesianVarianceModel input, EffectiveNumberOfParameters)
regressFullyDeterminedEMBayesianLinearModel initial_alpha initial_beta basis_fns ds
  = convergeOnEMBayesianLinearModel loopWorker design_matrix initial_alpha initial_beta basis_fns ds
  where
    n = fromIntegral $ dataSetSize ds
    m = fromIntegral $ length basis_fns
    
    design_matrix = regressDesignMatrix basis_fns (ds_inputs ds)
    
    -- In the limit n >> m, n - gamma = n, so we use that as the beta numerator
    -- We assume all paramaters are determined because n >> m, so we return m as gamma
    loopWorker _ _ = (n, m)
    
convergeOnEMBayesianLinearModel
    :: (Vectorable input)
    => (Precision -> Precision -> (Double, EffectiveNumberOfParameters)) -- ^ Loop worker: given alpha and beta, return new beta numerator and gamma
    -> Matrix Double        -- ^ Design matrix
    -> Precision            -- ^ Initial alpha
    -> Precision            -- ^ Initial beta
    -> [input -> Target]    -- ^ Basis functions
    -> DataSet
    -> (LinearModel input, BayesianVarianceModel input, EffectiveNumberOfParameters)
convergeOnEMBayesianLinearModel loop_worker design_matrix initial_alpha initial_beta basis_fns ds
  = loop eps initial_alpha initial_beta False
  where
    loop threshold alpha beta done
      | done      = (linear_model, BayesianVarianceModel { bvm_basis_fns = basis_fns, bvm_inv_hessian = inv_hessian, bvm_beta = beta }, gamma)
      | otherwise = loop (threshold * 2) alpha' beta' (eqWithin threshold alpha alpha' && eqWithin threshold beta beta')
      where
        (weights, inv_hessian) = bayesianPosteriorParameters alpha beta design_matrix (ds_targets ds)
        linear_model = LinearModel { lm_basis_fns = basis_fns, lm_weights = weights }
        
        (beta_numerator, gamma) = loop_worker alpha beta
        
        alpha' = gamma / (weights <.> weights)
        beta' = beta_numerator / modelSumSquaredError linear_model ds