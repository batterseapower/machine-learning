{-# LANGUAGE PatternSignatures #-}

--module Algorithms.MachineLearning.LinearRegression where
module Main where

import Numeric.LinearAlgebra

import GNUPlot
import System.Cmd

type Weight = Double


-- Description: Pairs (x,y) where:
--  x = U(0, 1)
--  y = sin(2πx) + N(0, 0.3)
-- Source: http://research.microsoft.com/~cmbishop/PRML/webdatasets/curvefitting.txt
sinData :: [(Double, Double)]
sinData = [
    (0.000000, 0.349486),
    (0.111111, 0.830839),
    (0.222222, 1.007332),
    (0.333333, 0.971507),
    (0.444444, 0.133066),
    (0.555556, 0.166823),
    (0.666667, -0.848307),
    (0.777778, -0.445686),
    (0.888889, -0.563567),
    (1.000000, 0.261502)
  ]

square :: Num a => a -> a
square x = x * x

singleton :: a -> [a]
singleton x = [x]

mean :: [Double] -> Double
mean xs = sum xs / (fromIntegral $ length xs)

void :: Monad m => m a -> m ()
void ma = ma >> return ()

normalGaussianBasis :: Double -> Double -> Double -> Double
normalGaussianBasis mean scale x = exp (negate $ (square (x - mean)) / (2 * (square scale)))

rationalToDouble :: Rational -> Double
rationalToDouble = realToFrac

basisFunctions :: [Double -> Double]
basisFunctions = const 1 : map (\mean -> normalGaussianBasis (rationalToDouble mean) 0.2) [-1,-0.9..1]

sumOfSquaresError :: [(Double, Double)] -> Double
sumOfSquaresError targetsAndPredictions = sum $ map (abs . uncurry (-)) targetsAndPredictions

predict :: [Weight] -> Double -> Double
predict weight x = sum $ zipWith (*) weight (map ($ x) basisFunctions)

solve :: [(Double, Double)] -> [Weight]
solve the_data = map head $ toLists $ pinv designMatrix `multiply` targetMatrix
  where
    (xs, targets) = unzip the_data
    
    targetMatrix = fromLists (map singleton targets)
    
    rows = map (\x -> map ($ x) basisFunctions) xs
    designMatrix = fromLists rows

evaluate :: [(Double, Double)] -> [Weight] -> IO ()
evaluate true_data weights = do
    putStrLn $ "Target Mean = " ++ show (mean (map snd true_data))
    putStrLn $ "Error = " ++ show (sumOfSquaresError comparable_data)
    putStrLn $ "Weights:\n" ++ show weights
    plot true_data (sample fittedFunction)
  where
    fittedFunction = predict weights
    comparable_data = map (\(x, true_target) -> (true_target, fittedFunction x)) true_data

sample :: (Double -> Double) -> [(Double, Double)]
sample f = map (\(x :: Rational) -> let x' = rationalToDouble x in (x', f x')) [0,0.01..1.0]

plot :: [(Double, Double)] -> [(Double, Double)] -> IO ()
plot true_samples fitted_samples = do
    plotPaths [EPS "output.ps"] [true_samples, fitted_samples]
    void $ rawSystem "open" ["output.ps"]

main :: IO ()
main = evaluate sinData weights
  where
    weights = solve sinData