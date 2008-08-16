-- | We commit the usual sin of lumping a load of useful functions with no clear
-- home in a "utilities" module
module Algorithms.MachineLearning.Utilities where

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