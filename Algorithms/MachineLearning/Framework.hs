{-# LANGUAGE FlexibleInstances #-}

module Algorithms.MachineLearning.Framework where

import Numeric.LinearAlgebra


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
        ds_targets :: Vector Double -- One row per sample, each value being a single target variable
    }

dataSetFromSampleList :: Vectorable a => [(a, Double)] -> DataSet a
dataSetFromSampleList elts
  = DataSet {
    ds_inputs = fromRows $ map (toVector . fst) elts,
    ds_targets = fromList $ map snd elts
  }

dataSetToSampleList :: Vectorable a => DataSet a -> [(a, Double)]
dataSetToSampleList ds = zip (map fromVector $ toRows $ ds_inputs ds) (toList $ ds_targets ds)

--
-- Models
--

class Model model where
    predict :: Vectorable input => model input -> input -> Double