module Main (main) where

import Algorithms.MachineLearning.BasisFunctions
import Algorithms.MachineLearning.Framework
import Algorithms.MachineLearning.LinearAlgebra
import Algorithms.MachineLearning.LinearClassification
import Algorithms.MachineLearning.LinearRegression
import Algorithms.MachineLearning.Tests.Data
import Algorithms.MachineLearning.Utilities

import GNUPlot

import Data.List
import Data.Ord

import System.Cmd
import System.Random


basisFunctions :: [Double -> Double]
basisFunctions 
  -- = [constantBasis]
  = gaussianBasisFamily (map rationalToDouble [-1,-0.5..1]) 0.09
  -- = gaussianBasisFamily (map rationalToDouble [-1,-0.9..1]) 0.04

basisFunctions2D :: [(Double, Double) -> Double]
basisFunctions2D 
  = [\(x, _) -> x, \(_, y) -> y]
  -- = map ((. \(x, y) -> fromList [x, y])) $ multivariateIsotropicGaussianBasisFamily [fromList [x, y] | x <- range, y <- range] 0.1
  where range = map rationalToDouble [-1,-0.8..1]

sampleFunction :: (Double -> a) -> [(Double, a)]
sampleFunction f = map (\(x :: Rational) -> let x' = rationalToDouble x in (x', f x'))
                       [0,0.01..1.0]

sampleFunction2D :: ((Double, Double) -> a) -> [((Double, Double), a)]
sampleFunction2D f = map (\((x, y) :: (Rational, Rational)) -> let x' = rationalToDouble x; y' = rationalToDouble y in ((x', y'), f (x', y')))
                          [(x, y) | x <- [-1.0,-0.99..1.0], y <- [-1.0,-0.99..1.0]]

evaluate :: (Vectorable input, Vectorable target, Model model input target, MetricSpace target) => model -> DataSet input target -> IO ()
evaluate model true_data = do
    putStrLn $ "Target Raw Means = " ++ show (map vectorMean (toColumns $ ds_targets true_data))
    putStrLn $ "Error = " ++ show (modelSumSquaredError model true_data)

plot :: [[(Double, Target)]] -> IO ()
plot sampless = do
    plotPaths [EPS "output.ps"] (map (sortBy (comparing fst)) sampless)
    void $ rawSystem "open" ["output.ps"]

plotClasses :: [((Double, Double), Class)] -> IO ()
plotClasses classess = do
    let -- Utilize a hack to obtain color output :-)
        color_eps filename = [EPS filename, ColorBox (Just [";set terminal postscript enhanced color"])]
        red_cross_style   = (Points, CustomStyle [PointType 1])
        blue_circle_style = (Points, CustomStyle [PointType 6]) -- These are actually /green/ circles, but who's counting?
        generations = [ (red_cross_style,   [position | (position, RedCross)   <- classess])
                      , (blue_circle_style, [position | (position, BlueCircle) <- classess]) ]
    plot2dMultiGen (color_eps "output.ps") generations
    void $ rawSystem "open" ["output.ps"]


linearModelTest :: IO ()
linearModelTest = do
    -- Do the regression
    let used_data = sinDataSet
        model = regressLinearModel basisFunctions used_data
    
    -- Show some model statistics
    evaluate model used_data
    putStrLn $ "Model For Target:\n" ++ show model
    
    -- Show some graphical information about the model
    plot [dataSetToSampleList used_data, sampleFunction $ predict model]

bayesianLinearModelTest :: IO ()
bayesianLinearModelTest = do
    gen <- newStdGen
    let used_data = sampleDataSet gen 10 sinDataSet
        (model, variance_model) = regressBayesianLinearModel 1 (1 / 0.09) basisFunctions used_data
    
    -- Show some model statistics
    evaluate model used_data
    putStrLn $ "Model For Target:\n" ++ show model
    putStrLn $ "Model For Variance:\n" ++ show variance_model
    
    -- Show some graphical information about the model
    plot [dataSetToSampleList used_data, sampleFunction $ predict model, sampleFunction $ (sqrt . predict variance_model)]

emBayesianLinearModelTest :: IO ()
emBayesianLinearModelTest = do
    gen <- newStdGen
    let used_data = sampleDataSet gen 10 sinDataSet
        (model, variance_model, gamma) = regressEMBayesianLinearModel 1 (1 / 0.09) basisFunctions used_data
    
    -- Show some model statistics
    evaluate model used_data
    putStrLn $ "Model For Target:\n" ++ show model
    putStrLn $ "Model For Variance:\n" ++ show variance_model
    putStrLn $ "Gamma = " ++ show gamma
    
    -- Show some graphical information about the model
    plot [dataSetToSampleList used_data, sampleFunction $ predict model, sampleFunction $ (sqrt . predict variance_model)]

linearClassificationModelTest :: IO ()
linearClassificationModelTest = do
    let used_data = classificationDataSet
        model = regressLinearClassificationModel basisFunctions2D used_data
    
    -- Show some model statistics
    evaluate model used_data
    
    -- Show some graphical information about the model
    plotClasses (dataSetToSampleList classificationDataSet)
    plotClasses (sampleFunction2D $ predict model)

main :: IO ()
main = linearClassificationModelTest