-- | Linear algebra used in the machine learning library: just HMatrix re-exports and some
-- other useful functions I have built up.
module Algorithms.MachineLearning.LinearAlgebra (
        module Numeric.LinearAlgebra,
        module Algorithms.MachineLearning.LinearAlgebra
    ) where

import Numeric.LinearAlgebra


-- | Given the input functions:
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

-- | Given the input functions:
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

-- | Summation of the elements in a vector.
vectorSum :: Element a => Vector a -> a
vectorSum v = constant 1 (dim v) <.> v

-- | The sum of the squares of the elements of the vector
vectorSumSquares :: Element a => Vector a -> a
vectorSumSquares v = v <.> v

-- | Mean of the elements in a vector.
vectorMean :: Element a => Vector a -> a
vectorMean v = (vectorSum v) / fromIntegral (dim v)

-- | Column-wise summation of a matrix.
sumColumns :: Element a => Matrix a -> Vector a
sumColumns m = constant 1 (rows m) <> m

-- | Create a constant matrix of the given dimension, analagously to 'constant'.
constantM :: Element a => a -> Int -> Int -> Matrix a
constantM elt row_count col_count = reshape row_count (constant elt (row_count * col_count))

matrixToVector :: Element a => Matrix a -> Vector a
matrixToVector m
  | rows m == 1  -- Row vector
  || cols m == 1 -- Column vector
  = flatten m
  | otherwise
  = error "matrixToVector: matrix is neither a row or column vector"

trace :: Element a => Matrix a -> a
trace = vectorSum . takeDiag