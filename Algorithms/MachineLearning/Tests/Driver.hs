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
basisFunctions = gaussianBasisFamily (map rationalToDouble [-1,-0.9..1]) 0.04

sumOfSquaresError :: [(Double, Double)] -> Double
sumOfSquaresError targetsAndPredictions = sum $ map (abs . uncurry (-)) targetsAndPredictions

sampleFunction :: (Double -> Double) -> [(Double, Double)]
sampleFunction f = map (\(x :: Rational) -> let x' = rationalToDouble x in (x', f x')) [0,0.01..1.0]

evaluate :: (Model model, Show (model Double)) => model Double -> DataSet -> IO ()
evaluate model true_data = do
    putStrLn $ "Target Mean = " ++ show (vectorMean (ds_targets true_data))
    putStrLn $ "Error = " ++ show (modelSumSquaredError model true_data)

plot :: [[(Double, Target)]] -> IO ()
plot sampless = do
    plotPaths [EPS "output.ps"] (map (sortBy (comparing fst)) sampless)
    void $ rawSystem "open" ["output.ps"]

main :: IO ()
main = do
    gen <- newStdGen
    let --used_data = sinDataSet
        used_data = sampleDataSet gen 5 sinDataSet
        (model, variance_model, gamma) = regressEMBayesianLinearModel 5 (1 / 0.3) basisFunctions used_data
    
    -- Show some model statistics
    evaluate model used_data
    putStrLn $ "Model For Target:\n" ++ show model
    putStrLn $ "Model For Variance:\n" ++ show variance_model
    putStrLn $ "Gamma = " ++ show gamma
    
    -- Show some graphical information about the model
    plot [dataSetToSampleList used_data, sampleFunction $ predict model, sampleFunction $ predict variance_model]
