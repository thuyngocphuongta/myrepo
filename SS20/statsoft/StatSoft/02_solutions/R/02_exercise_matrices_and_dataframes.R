
# This exercise is concerned with basic operations on matrices and data frames.
# We are using only methods contained in the standard R installation; don't use
# tidy here.


# Given a square matrix as input, return the values on the "counter diagonal" going
# from top left to bottom right. E.g. for
#
# 1 2 3
# 4 5 6 --> c(3, 5, 7).
# 7 8 9
#
# Make sure this works with the trivial matrix with only one element!
ex01Counterdiagonal <- function(mat) {
  # your code
  diag(mat[, rev(seq_len(ncol(mat))), drop = FALSE])
}

# Given a matrix as input, return the coordinates (row and column) of the maximum element,
# as a two element numeric vector. If multiple elements are tied for maximum, you
# can return the coordinates of any of them.
#
# E.g.
#
# 1 2 3
# 4 5 6 --> c(3, 3)
# 7 8 9
#
# 0 0
# 0 0 --> c(3, 1)
# 3 0
#
# 1 0 --> both c(1, 1) and c(2, 2) would be correct
# 0 1
ex02MatrixWhichMax <- function(mat) {
  # your code
  wm <- which.max(mat)
  as.numeric(c(row(mat)[wm], col(mat)[wm]))
}

# Given a `character` vector of sentences, create a matrix containing a
# "bag-of-words"-Representation. E.g. consider you get the input
#
# > c("THE HOUSE WAS A SMALL HOUSE", "GERHARD WAS IN A HOUSE", "GERHARD WAS A SMALL MAN")
#
# the bag-of-words representation counts how many times each words occurs in each of
# the elements. It is a matrix containing one column for each word found in any of
# the strings, and a row for each of the strings, counting how often this word occurs
# in this string. In the example above, it would be
#
# THE      HOUSE   WAS     A       SMALL    GERHARD IN       MAN
# 1        2       1       1       1        0       0        0
# 0        1       1       1       0        1       1        0
# 0        0       1       1       1        1       0        1
#
# The order of the columns does not matter, but the `colnames` of this matrix must
# be the words they represent. You can rely on the fact that words consist of only upper-case
# letters, that they are always separated by exactly one space, and that no other special
# characters or symbols occur. You may find the `strsplit()` and the `table()` function helpful.
ex03BagOfWords <- function(sentences) {
  # your code
  split <- strsplit(sentences, " ")
  allwords <- unique(unlist(split))
  split <- lapply(split, function(sen) {
    table(factor(sen, levels = allwords))
  })
  do.call(rbind, split)
}


# Your function is given a `data.frame` with columns that can each be of any of numeric, logical
# or factor type. You are supposed to return a `data.frame` containing only columns selected
# by type. For this you are given the "type" argument: a character vector containing a (possibly improper)
# subset of the values c("numeric", "logical", "factor").
#
# E.g. * data = iris (the "iris" dataset as contained in R)
#        type = "numeric"
#        --> return
#                     Sepal.Length Sepal.Width Petal.Length Petal.Width
#                   1          5.1         3.5          1.4         0.2
#                   2          4.9         3.0          1.4         0.2
#                   3          4.7         3.2          1.3         0.2
#                   .........
#      * data = iris
#        type = "factor"
#        --> return
#                     Species
#                   1  setosa
#                   2  setosa
#                   3  setosa
#      * data = iris
#        type = "logical"
#        --> return a data frame with 0 columns and 150 rows
#      * data = iris
#        type = c("factor", "numeric")
#    or: type = c("numeric", "factor")
#        --> return the original iris data
#    ... etc.
# The order of all included columns should not be changed with respect to their order in the input, and
# should be independent of the order in the `type` argument.
# You may assume that there are no columns that are not one of numeric, logical, or factor.
ex04SelectDF <- function(data, type) {
  # your code
  sel.col <- vapply(data, function(col) {
    any(type %in% class(col))
  }, TRUE)
  data[sel.col]
}


# You are given a `data.frame` with some data about study participants. The first column is "sex", a
# factor variable with levels c("male", "female"). It is followed by further columns, all of which
# are numeric, and some of which have missing values (NA). Your task is to *impute* the missing values,
# i.e. set them to putative numeric values inferred from the other participants. In particular,
# you are to set the missing values in each variable to the *mean value* of the *non-missing values*
# the variable *within the same sex group*.
#
# E.g. for the input
# > data.frame(
#     sex = factor(c("male", "male", "male", "female", "female", "female")),
#     height = c(178, 185, NA, 157, NA, 174), weight = c(95, 90, 99, 70, 77, NA),
#     age = c(23, NA, NA, NA, 21, 22))
# the returned value should be
# > data.frame(
#     sex = factor(c("male", "male", "male", "female", "female", "female")),
#     height = c(178, 185, 181.5, 157, 165.5, 174), weight = c(95, 90, 99, 70, 77, 73.5),
#     age = c(23, 23, 23, 21.5, 21, 22))
#
# The return value should be a data.frame with the same columns as the input and with the missing values
# imputed. You can assume that there is at least one value present for each sex group.
# However, you can *not* assume that males and females are blocked together, i.e.
# the "sex" feature could also be factor(c("male", "female", "male", "female", "female", "male")).
# Your input may have different column names, but the first column will be named "sex".
ex05Imputation <- function(data) {
  # your code
  for (cn in setdiff(colnames(data), "sex")) {
    col <- data[[cn]]
    data[[cn]][is.na(col)] <- tapply(col, data$sex, mean, na.rm = TRUE)[data$sex[is.na(col)]]
  }
  data
}
