module Algorithms.MachineLearning.LinearClassification (
        DiscriminantModel,
        regressLinearClassificationModel,
    ) where

import Algorithms.MachineLearning.Framework
import Algorithms.MachineLearning.LinearAlgebra
import Algorithms.MachineLearning.LinearRegression
import Algorithms.MachineLearning.Utilities

import Data.Ord
import Data.List
import Data.Maybe


data (Bounded classes, Enum classes) => DiscriminantModel input classes = DiscriminantModel {
        dm_class_models :: AnyModel input (Vector Double)
    }

instance (Vectorable input, Bounded classes, Enum classes) => Model (DiscriminantModel input classes) input classes where
    predict model input = snd $ maximumBy (comparing fst) predictions
      where
        predictions = toList (predict (dm_class_models model) input) `zip` enumAsList


regressLinearClassificationModel :: (Vectorable input, Vectorable classes, Bounded classes, Enum classes, Eq classes)
                                 => [input -> Double]     -- ^ Basis functions
                                 -> DataSet input classes -- ^ Class mapping to use for training
                                 -> DiscriminantModel input classes
regressLinearClassificationModel basis_fns ds = DiscriminantModel { dm_class_models = class_models }
  where
    class_models = AnyModel $ regressLinearModel basis_fns (fmapDataSetTarget classToCharacteristicVector ds)
    indexed_classes = enumAsList `zip` [0..]
    classToCharacteristicVector the_class = fromList $ replicate (index - 1) 0 ++ [1] ++ replicate (size - index - 1) 0
      where
        size = enumSize the_class
        index = fromJust $ lookup the_class indexed_classes