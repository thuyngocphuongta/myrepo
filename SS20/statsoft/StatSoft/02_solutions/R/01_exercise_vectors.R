
# This exercise is concerned with vectors and basic operations on them.
# Solve this exercises by filling in the function bodies, and try to read up no
# things that you didn't know or surprise you.

# write a function that divides x by y
# x and y are both single numbers.
ex01Divide <- function(x, y) {
  # your code
  x / y
}

# write a function that divides x by y
# x and y are numeric vectors of the same length. Does this function need to
# differ from the one above?
ex02DivideVectors <- function(x, y) {
  # your code
  x / y
}

# write a function that checks if a "logical", a "numeric", or a "character" vector or something else.
# the function should return "L" if the input is logical, "N" if it is numeric, "C" for characters, and otherwise "X".
ex03VectorType <- function(x) {
  # your code
  switch(mode(x), numeric = "N", logical = "L", character = "C", "X")
}

# write a function that takes a vector of integers and returns all numbers that are even.
# an empty vector should be returned when no even numbers are found.
ex04Even <- function(x) {
  # your code
  x[x %% 2 == 0]
}

# Suppose you have a vector "numbers" of ages of people, and you want to know how many of them are "Youth" (younger
# than 18 years), "Young Adult" (at least 18, less than 36), "Adult" (at least 36, less than 56) or "Senior" (56 and
# up). Or another problem: you have a vector of people's BMI and you want to know how many are "Severely underweight"
# (BMI < 16), "Underweight" (BMI between 16 and 18.5), "Healthy" (18.5 to 25) or "Overweight" (BMI > 25).
#
# For these problems, you get as input a vector of numbers, and a vector of bin cutoffs, the cutoff vector being a
# *named* vector, indicating the upper bounds of each bin. The upper bound of the last bin is `Inf` (infinity). Your
# task is to count the numbers in each bin and return them in a named vector, in order of increasing cutoff value.
#
# The inputs for the first problem described above could be
# > numbers = c(10.3, 32.7, 50.5, 62.4, 32.0, 50.4, 19.7, 60.1, 69.0, 50.6, 11.1, 48.6, 17.4, 34.3, 78.7)
# > cutoffs = c(Youth = 18, "Young Adult" = 36, Adult = 56, Senior = Inf)
# and the expected result would be
# > c(Youth = 3, "Young Adult" = 4, Adult = 4, Senior = 4)
# The second problem could have input
# > numbers = c(20, 23, 28)
# > cutoffs = c("Severely underweight" = 16, Underweight = 18.5, Healthy = 25, Overweight = Inf)
# and the result should be
# > c("Severely underweight" = 0, Underweight = 0, Healthy = 2, Overweight = 1)
#
# You may assume that the cutoff values are already sorted and that the largest upper bound is always `Inf`.
# Cutoff bounds are exclusive, so a data point of 18.0 in the first example is counted as "Young Adult".
#
# There are different ways to solve this; among others you could solve this task by itself, you could also solve the
# following exercise (ex06Binning) first and call it here using the `table()` method, or you could
# try if the `hist()` function is useful.
ex05BinCounting <- function(numbers, cutoffs) {
  # your code
  cumulative <- vapply(cutoffs, function(x) sum(numbers < x), 0)
  diff(c(0, cumulative))
}

# Related to the last exercise: Now you should not only *count* the number of data points in each bin, you
# should return, for each data point, the bin that it belongs to, as an *ordered factor* variable. This iscalled data
# binning. For the first input, this would be
# > numbers = c(10.3, 32.7, 50.5, 62.4, 32.0, 50.4, 19.7, 60.1, 69.0, 50.6, 11.1, 48.6, 17.4, 34.3, 78.7)
# > cutoffs = c(Youth = 18, "Young Adult" = 36, Adult = 56, Senior = Inf)
# return:
# > ordered(c("Youth", "Young Adult", "Adult", "Senior", "Young Adult", "Adult",  "Young Adult", "Senior",
#             "Senior", "Adult", "Youth", "Adult",  "Youth", "Young Adult", "Senior"),
#           levels = c("Youth", "Young Adult", "Adult", "Senior"))
# The second problem would be
# > numbers = c(20, 23, 28)
# > cutoffs = c("Severely underweight" = 16, Underweight = 18.5, Healthy = 25, Overweight = Inf)
# and the result should be
# > ordered(c("Healthy", "Healthy", "Overweight"),
#   levels = c("Severely underweight", "Underweight", "Healthy", "Overweight"))
#
# The `cut()` function could be helpful here.
ex06Binning <- function(numbers, cutoffs) {
  # your code
  cut(numbers, breaks = c(-Inf, cutoffs), labels = names(cutoffs), right = FALSE, ordered_result = TRUE)
}


# "Fizz Buzz" is a game for children where players take turns saying numbers
# counting up, but say "Fizz" instead if a number is divisible by 3 and "Buzz"
# if a number is divisible by 5, saying "Fizz Buzz" if both is the case.
# Read about it at Wikipdia: <https://en.wikipedia.org/wiki/Fizz_buzz>.
#
# Write a function that plays this game with itself. The function should
# return a `character` vector containing the numbers up to `up.to`, or
# "Fizz", "Buzz" or "Fizz Buzz" at the appropriate places. The numbers at which
# "Fizz" and "Buzz" should be returned are optional arguments.
#
# Example:
# up.to = 10, fizz.number = 3, buzz.number = 5
# --> returns c("1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz")
# up.to = 6, fizz.number = 2, buzz.number = 4
# --> returns c("1", "Fizz", "3", "Fizz Buzz", "5", "Fizz")
#
# You can rely on fizz.number, buzz.number and up.to being integer values greater
# than 0, though not necessarily greater than 1 and not necessarily different from
# each other.
#
# Although you are allowed to use loops, it is probably simpler to solve this
# without loops.
ex07FizzBuzz <- function(up.to, fizz.number = 3, buzz.number = 5) {
  # your code
  sequence <- seq_len(up.to)
  ret <- as.character(sequence)
  is.fizz <- sequence %% fizz.number == 0
  is.buzz <- sequence %% buzz.number == 0
  ret[is.fizz] <- "Fizz"
  ret[is.buzz] <- "Buzz"
  ret[is.fizz & is.buzz] <- "Fizz Buzz"
  ret
}
