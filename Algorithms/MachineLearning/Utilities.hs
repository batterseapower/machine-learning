-- | We commit the usual sin of lumping a load of useful functions with no clear
-- home in a "utilities" module
module Algorithms.MachineLearning.Utilities where

import Data.List
import Data.Ord

import System.Random


newtype K a = K { unK :: a }

instance Functor K where
    fmap f (K x) = K (f x)


square :: Num a => a -> a
square x = x * x

singleton :: a -> [a]
singleton x = [x]

void :: Monad m => m a -> m ()
void ma = ma >> return ()

rationalToDouble :: Rational -> Double
rationalToDouble = realToFrac

onLeft :: (a -> c) -> (a, b) -> (c, b)
onLeft f (x, y) = (f x, y)

onRight :: (b -> c) -> (a, b) -> (a, c)
onRight f (x, y) = (x, f y)

shuffle :: StdGen -> [a] -> [a]
shuffle gen xs = map snd $ sortBy (comparing fst) (zip (randoms gen :: [Double]) xs)

chunk :: Int -> [a] -> [[a]]
chunk _ [] = []
chunk n xs = this : chunk n rest
  where
    (this, rest) = splitAt n xs

sample :: StdGen -> Int -> [a] -> [a]
sample gen n xs = take n (shuffle gen xs)

eqWithin :: Double -> Double -> Double -> Bool
eqWithin jitter left right = abs (left - right) < jitter

enumAsList :: (Enum a, Bounded a) => [a]
enumAsList = enumFromTo minBound maxBound

enumSize :: (Enum a, Bounded a) => a -> Int
enumSize what_enum = length (enumAsList `asTypeOf` [what_enum])