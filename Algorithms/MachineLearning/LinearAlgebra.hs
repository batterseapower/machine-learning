module Algorithms.MachineLearning.LinearAlgebra (
        module Numeric.LinearAlgebra,
        module Algorithms.MachineLearning.LinearAlgebra
    ) where

import Numeric.LinearAlgebra


-- Given the input functions:
--
-- @[f_1, f_2, ..., f_n]@
--
-- and input matrix:
--
-- @
--  [ r_1
--  , r_2
--  , ...
--  , r_m ]
-- @
--
-- returns the output matrix:
--
-- @
--  [ f_1(r_1), f_2(r_1), ..., f_n(r_1)
--  , f_1(r_2), f_2(r_2), ..., f_n(r_2)
--  , f_1(r_m), f_2(r_m), ..., f_n(r_m) ]
-- @
applyMatrix :: [Vector Double -> Double] -> Matrix Double -> Matrix Double
applyMatrix fns inputs = fromLists (map (\r -> map ($ r) fns) rs)
  where
    rs = toRows inputs

-- Given the input functions:
--
-- @[f_1, f_2, ..., f_n]@
--
-- and input:
--
-- @x@
--
-- returns the output vector:
--
-- @
--  [ f_1(x)
--  , f_2(x)
--  , ...
--  , f_n(x) ]
-- @
applyVector :: [inputs -> Double] -> inputs -> Vector Double
applyVector fns inputs = fromList $ map ($ inputs) fns

-- Summation of a vector
vectorSum :: Vector Double -> Double
vectorSum v = constant 1 (dim v) <.> v

-- Vector mean
vectorMean :: Vector Double -> Double
vectorMean v = (vectorSum v) / fromIntegral (dim v)

-- Column-wise summation of the matrix
sumColumns :: Matrix Double -> Vector Double
sumColumns m = constant 1 (rows m) <> m