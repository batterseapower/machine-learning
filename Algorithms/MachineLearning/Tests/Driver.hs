module Main where

import Algorithms.MachineLearning.BasisFunctions
import Algorithms.MachineLearning.Framework
import Algorithms.MachineLearning.LinearAlgebra
import Algorithms.MachineLearning.LinearRegression
import Algorithms.MachineLearning.Tests.Data
import Algorithms.MachineLearning.Utilities

import GNUPlot

import Data.List
import Data.Ord

import System.Cmd
import System.Random


basisFunctions :: [Double -> Double]
basisFunctions = const 1 : map (\mean -> gaussianBasis (rationalToDouble mean) 0.04) [-1,-0.9..1]

sumOfSquaresError :: [(Double, Double)] -> Double
sumOfSquaresError targetsAndPredictions = sum $ map (abs . uncurry (-)) targetsAndPredictions

sample :: (Double -> Double) -> [(Double, Double)]
sample f = map (\(x :: Rational) -> let x' = rationalToDouble x in (x', f x')) [0,0.01..1.0]

evaluate :: (Model model, Show (model Double)) => model Double -> DataSet -> IO ()
evaluate model true_data = do
    putStrLn $ "Target Mean = " ++ show (vectorMean (ds_targets true_data))
    putStrLn $ "Error = " ++ show (sumOfSquaresError comparable_data)
    putStrLn $ "Model:\n" ++ show model
  where
    fittedFunction = predict model
    comparable_data = map (fittedFunction `onLeft`) (dataSetToSampleList true_data)

plot :: [[(Double, Target)]] -> IO ()
plot sampless = do
    plotPaths [EPS "output.ps"] (map (sortBy (comparing fst)) sampless)
    void $ rawSystem "open" ["output.ps"]

main :: IO ()
main = do
    gen <- newStdGen
    let used_data = head $ binDS gen 2 sinDataSet
        (model, variance_model, gamma) = regressEMBayesianLinearModel 5 (1 / 0.3) basisFunctions used_data
    
    -- Show some model statistics
    evaluate model used_data
    print $ "Gamma = " ++ show gamma
    
    -- Show some graphical information about the model
    plot [dataSetToSampleList used_data, sample $ predict model, sample $ predict variance_model]
