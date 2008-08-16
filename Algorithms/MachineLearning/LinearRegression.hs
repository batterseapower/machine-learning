{-# LANGUAGE PatternSignatures, RecordPuns #-}

module Algorithms.MachineLearning.LinearRegression where

import Algorithms.MachineLearning.Framework
import Algorithms.MachineLearning.LinearAlgebra


type Weight = Double

data LinearModel input = LinearModel {
        lm_basis_fns :: [input -> Double],
        lm_weights   :: Vector Double
    }

instance Show (LinearModel input) where
    show model = show (lm_weights model)

instance Model LinearModel where
    predict (LinearModel { lm_basis_fns, lm_weights }) input = lm_weights <.> phi_app_x
      where
        phi_app_x = applyVector lm_basis_fns input

regressLinearModel :: (Vectorable input) => [input -> Double] -> DataSet input -> LinearModel input
regressLinearModel basis_fns (DataSet { ds_inputs, ds_targets })
  = LinearModel { lm_basis_fns = basis_fns, lm_weights = weights }
  where
    designMatrix = applyMatrix (map (. fromVector) basis_fns) ds_inputs -- One row per sample, one column per basis function
    weights = pinv designMatrix <> ds_targets

