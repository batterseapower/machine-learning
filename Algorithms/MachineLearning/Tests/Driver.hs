module Main where

import Algorithms.MachineLearning.Framework
import Algorithms.MachineLearning.LinearAlgebra
import Algorithms.MachineLearning.LinearRegression
import Algorithms.MachineLearning.Tests.Data
import Algorithms.MachineLearning.Utilities

import GNUPlot
import System.Cmd


normalGaussianBasis :: Double -> Double -> Double -> Double
normalGaussianBasis mean scale x = exp (negate $ (square (x - mean)) / (2 * (square scale)))

basisFunctions :: [Double -> Double]
basisFunctions = const 1 : map (\mean -> normalGaussianBasis (rationalToDouble mean) 0.2) [-1,-0.9..1]

sumOfSquaresError :: [(Double, Double)] -> Double
sumOfSquaresError targetsAndPredictions = sum $ map (abs . uncurry (-)) targetsAndPredictions

sample :: (Double -> Double) -> [(Double, Double)]
sample f = map (\(x :: Rational) -> let x' = rationalToDouble x in (x', f x')) [0,0.01..1.0]

evaluate :: (Model model, Show (model Double)) => model Double -> DataSet Double -> IO ()
evaluate model true_data = do
    putStrLn $ "Target Mean = " ++ show (vectorMean (ds_targets true_data))
    putStrLn $ "Error = " ++ show (sumOfSquaresError comparable_data)
    putStrLn $ "Model:\n" ++ show model
    plot (dataSetToSampleList true_data) (sample fittedFunction)
  where
    fittedFunction = predict model
    comparable_data = map (fittedFunction `onLeft`) (dataSetToSampleList true_data)

plot :: [(Double, Target)] -> [(Double, Target)] -> IO ()
plot true_samples fitted_samples = do
    plotPaths [EPS "output.ps"] [true_samples, fitted_samples]
    void $ rawSystem "open" ["output.ps"]

main :: IO ()
main = evaluate (regressLinearModel basisFunctions sinDataSet) sinDataSet