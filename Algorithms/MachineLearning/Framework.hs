{-# LANGUAGE FlexibleInstances #-}

-- | The "framework" provides all the core classes and types used ubiquitously by
-- the machine learning algorithms.
module Algorithms.MachineLearning.Framework where

import Numeric.LinearAlgebra


--
-- Ubiquitous synonyms for documentation purposes
--

type Target = Double
type Weight = Double

-- | A positive constant indicating how strongly regularization should be applied. A good
-- choice for this parameter might be your belief about the variance of the inherent noise
-- in the samples (1/beta) divided by your belief about the variance of the weights that
-- should be learnt by the model (1/alpha).
--
-- See also equation 3.55 and 3.28 in Bishop.
type RegularizationCoefficient = Double

--
-- Injections to and from vectors
--
class Vectorable a where
    toVector :: a -> Vector Double
    fromVector :: Vector Double -> a

instance Vectorable Double where
    toVector = flip constant 1
    fromVector = flip (@>) 0

instance Vectorable (Double, Double) where
    toVector (x, y) = 2 |> [x, y]
    fromVector vec = (vec @> 0, vec @> 1)

instance Vectorable (Vector Double) where
    toVector = id
    fromVector = id

--
-- Labelled data set
--

data DataSet input = DataSet {
        ds_inputs :: Matrix Double, -- One row per sample, one column per input variable
        ds_targets :: Vector Target -- One row per sample, each value being a single target variable
    }

dataSetFromSampleList :: Vectorable a => [(a, Target)] -> DataSet a
dataSetFromSampleList elts
  = DataSet {
    ds_inputs = fromRows $ map (toVector . fst) elts,
    ds_targets = fromList $ map snd elts
  }

dataSetToSampleList :: Vectorable a => DataSet a -> [(a, Target)]
dataSetToSampleList ds = zip (map fromVector $ toRows $ ds_inputs ds) (toList $ ds_targets ds)

--
-- Models
--

class Model model where
    predict :: Vectorable input => model input -> input -> Target