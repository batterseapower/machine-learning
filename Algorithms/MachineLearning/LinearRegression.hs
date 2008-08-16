{-# LANGUAGE PatternSignatures #-}

module Algorithms.MachineLearning.LinearRegression where

import Algorithms.MachineLearning.Framework
import Algorithms.MachineLearning.LinearAlgebra


data LinearModel input = LinearModel {
        lm_basis_fns :: [input -> Double],
        lm_weights   :: Vector Double
    }

instance Show (LinearModel input) where
    show model = show (lm_weights model)

instance Model LinearModel where
    predict model input = (lm_weights model) <.> phi_app_x
      where
        phi_app_x = applyVector (lm_basis_fns model) input

regressLinearModel :: (Vectorable input) => [input -> Double] -> DataSet input -> LinearModel input
regressLinearModel basis_fns ds
  = LinearModel { lm_basis_fns = basis_fns, lm_weights = weights }
  where
    designMatrix = applyMatrix (map (. fromVector) basis_fns) (ds_inputs ds) -- One row per sample, one column per basis function
    weights = pinv designMatrix <> (ds_targets ds)

