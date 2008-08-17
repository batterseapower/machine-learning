{-# LANGUAGE FlexibleInstances #-}

-- | The "framework" provides all the core classes and types used ubiquitously by
-- the machine learning algorithms.
module Algorithms.MachineLearning.Framework where

import Algorithms.MachineLearning.Utilities

import Numeric.LinearAlgebra

import Data.List

import System.Random

--
-- Ubiquitous synonyms for documentation purposes
--

-- | The target is the variable you wish to predict with your machine learning algorithm.
type Target = Double

type Weight = Double

-- | Commonly called the "average" of a set of data.
type Mean = Double

-- | Variance is the mean squared deviation from the mean.
type Variance = Double

-- | Precision is the inverse of variance.
type Precision = Double

-- | A positive coefficient indicating how strongly regularization should be applied. A good
-- choice for this parameter might be your belief about the variance of the inherent noise
-- in the samples (1/beta) divided by your belief about the variance of the weights that
-- should be learnt by the model (1/alpha).
--
-- Commonly written as lambda.
--
-- See also equation 3.55 and 3.28 in Bishop.
type RegularizationCoefficient = Double

-- | A positive number that indicates the number of fully determined parameters in a learnt
-- model.  If all your parameters are determined, it will be equal to the number of parameters
-- available, and if your data did not support any parameters it will be simply 0.
--
-- See also section 3.5.3 of Bishop.
type EffectiveNumberOfParameters = Double

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

data DataSet = DataSet {
        ds_inputs :: Matrix Double, -- One row per sample, one column per input variable
        ds_targets :: Vector Target -- One row per sample, each value being a single target variable
    }

dataSetFromSampleList :: Vectorable input => [(input, Target)] -> DataSet
dataSetFromSampleList elts
  = DataSet {
    ds_inputs = fromRows $ map (toVector . fst) elts,
    ds_targets = fromList $ map snd elts
  }

dataSetToSampleList :: Vectorable input => DataSet -> [(input, Target)]
dataSetToSampleList ds = zip (dataSetInputs ds) (dataSetTargets ds)

dataSetInputs :: Vectorable input => DataSet -> [input]
dataSetInputs ds = map fromVector $ toRows $ ds_inputs ds

dataSetTargets :: DataSet -> [Target]
dataSetTargets ds = toList $ ds_targets ds

binDS :: StdGen -> Int -> DataSet -> [DataSet]
binDS gen bins ds = map dataSetFromSampleList $ chunk (ceiling $ (fromIntegral $ length samples :: Double) / (fromIntegral bins)) shuffled_samples
  where
    samples = zip (toRows $ ds_inputs ds) (toList $ ds_targets ds)
    shuffled_samples = shuffle gen samples

--
-- Models
--

class Model model where
    predict :: Vectorable input => model input -> input -> Target

modelSumSquaredError :: (Model model, Vectorable input) => model input -> DataSet -> Double
modelSumSquaredError model ds = error_vector <.> error_vector
  where
    error_vector = ds_targets ds - fromList (map (predict model) (dataSetInputs ds))