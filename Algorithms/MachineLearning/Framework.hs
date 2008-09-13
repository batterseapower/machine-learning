-- | The "framework" provides all the core classes and types used ubiquitously by
-- the machine learning algorithms.
module Algorithms.MachineLearning.Framework where

import Algorithms.MachineLearning.LinearAlgebra
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

-- | Variance is the mean squared deviation from the mean. Must be positive.
type Variance = Double

-- | Precision is the inverse of variance. Must be positive.
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

data DataSet input target = DataSet {
        ds_inputs :: Matrix Double, -- One row per sample, one column per input variable
        ds_targets :: Matrix Target -- One row per sample, one column per target variable
    }

fmapDataSetInput :: (Vectorable input, Vectorable input', Vectorable target) => (input -> input') -> DataSet input target -> DataSet input' target
fmapDataSetInput f = dataSetFromSampleList . fmap (onLeft f) . dataSetToSampleList

fmapDataSetTarget :: (Vectorable input, Vectorable target, Vectorable target') => (target -> target') -> DataSet input target -> DataSet input target'
fmapDataSetTarget f = dataSetFromSampleList . fmap (onRight f) . dataSetToSampleList

dataSetFromSampleList :: (Vectorable input, Vectorable target) => [(input, target)] -> DataSet input target
dataSetFromSampleList elts
  | null elts = error "dataSetFromSampleList: no data supplied"
  | otherwise = DataSet {
    ds_inputs = fromRows $ map (toVector . fst) elts,
    ds_targets = fromRows $ map (toVector . snd) elts
  }

dataSetToSampleList :: (Vectorable input, Vectorable target) => DataSet input target -> [(input, target)]
dataSetToSampleList ds = zip (dataSetInputs ds) (dataSetTargets ds)

dataSetInputs :: Vectorable input => DataSet input target -> [input]
dataSetInputs ds = map fromVector $ toRows $ ds_inputs ds

dataSetTargets :: Vectorable target => DataSet input target -> [target]
dataSetTargets ds = map fromVector $ toRows $ ds_targets ds

dataSetInputLength :: DataSet input target -> Int
dataSetInputLength ds = cols (ds_inputs ds)

dataSetSize :: DataSet input target -> Int
dataSetSize ds = rows (ds_inputs ds)

binDataSet :: StdGen -> Int -> DataSet input target -> [DataSet input target]
binDataSet gen bins = transformDataSetAsVectors binDataSet'
  where
    binDataSet' ds = map dataSetFromSampleList $ chunk bin_size shuffled_samples
      where
        shuffled_samples = shuffle gen (dataSetToSampleList ds)
        bin_size = ceiling $ (fromIntegral $ dataSetSize ds :: Double) / (fromIntegral bins)

sampleDataSet :: StdGen -> Int -> DataSet input target -> DataSet input target
sampleDataSet gen n = unK . transformDataSetAsVectors (K . dataSetFromSampleList . sample gen n . dataSetToSampleList)

transformDataSetAsVectors :: Functor f => (DataSet (Vector Double) (Vector Double) -> f (DataSet (Vector Double) (Vector Double))) -> DataSet input target -> f (DataSet input target)
transformDataSetAsVectors transform input = fmap castDataSet (transform (castDataSet input))
  where
    castDataSet :: DataSet input1 target1 -> DataSet input2 target2
    castDataSet ds = DataSet {
          ds_inputs = ds_inputs ds,
          ds_targets = ds_targets ds
      }

--
-- Metric spaces
--

class MetricSpace a where
    distance :: a -> a -> Double

instance MetricSpace Double where
    distance x y = abs (x - y)

instance MetricSpace (Vector Double) where
    distance x y = vectorSumSquares (x - y)

--
-- Models
--

class Model model input target | model -> input target where
    predict :: model -> input -> target

data AnyModel input output = forall model. Model model input output => AnyModel { theModel :: model }

instance Model (AnyModel input output) input output where
    predict (AnyModel model) = predict model

modelSumSquaredError :: (Model model input target, MetricSpace target, Vectorable input, Vectorable target) => model -> DataSet input target -> Double
modelSumSquaredError model ds = sum [sample_error * sample_error | sample_error <- sample_errors]
  where
    sample_errors = zipWith (\x y -> x `distance` y) (dataSetTargets ds) (map (predict model) (dataSetInputs ds))