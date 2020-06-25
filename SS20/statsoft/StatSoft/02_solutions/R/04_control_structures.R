
# This exercise is concerned with control structures: conditional execution and loops.
# (we will skip functions for now).

# A "Semantic Network" represents knowledge about real-world relationships in
# computer-readable form. This function will use a database containing "is-a"
# relationships and make inference about the objects. An example knowledge database
# has the form
#
#         entity               is.a
#            cat             mammal
#            dog             mammal
#            dog  man's best friend
#         mammal             animal
#  felix the cat                cat
# indicating that "felix the cat" is a "cat", and that a "cat" is a "mammal" and so on.
# The function should now take two entities, `special` and `general` and return
# TRUE whenever `special` "is a" `general` according to the database, FALSE othwerise.
# Example inputs using the the knowledge database from above:
# > knowledge = data.frame(entity = c("cat", "dog", "dog", "mammal", "felix the cat"),
#     is.a = c("mammal", "mammal", "man's best friend", "animal", "cat"), stringsAsFactors = FALSE)
# The following are TRUE:
# > special = "cat"
# > general = "animal"
# --> TRUE
# > special = "felix the cat"
# > general = "animal"
# --> TRUE
# > special = "dog"
# > general = "dog"
# --> TRUE  # !!!
# The following default to FALSE:
# > special = "cat"
# > general = "dog"
# --> FALSE
# > special = "animal"
# > general = "cat"
# --> FALSE
# > special = "animal"
# > general = "mammal"
# --> FALSE
# > special = "felix the cat"
# > general = "dog"
# --> FALSE
#
# You may assume there are no circular relationships. You may *not* assume that
# `special` or `general` actually occur in the knowledge database. If any of them
# do not occur, the result is FALSE, except if both `special` and `general` are the
# same:
# > special = "bird"
# > general = "animal"
# --> FALSE  # bird not known to database
# > special = "bird"
# > general = "bird"
# --> TRUE  # a bird is a bird
ex01SemanticNetwork <- function(knowledge, special, general) {
  # your code

  # the solution checks if the "general" item is either equal to "special",
  # or equal to any of the things that are `generalizations` of "special"
  # according to the knowledge database.
  while (length(special)) {
    if (general %in% special) return(TRUE)
    special <- knowledge$is.a[which(knowledge$entity %in% special)]
  }
  FALSE
}

# A "cellular automaton" is a discrete model of state evolution. We consider a state
# of a vector with "cells" of values 0 or 1, for example the vector c(0, 0, 0, 1, 0, 0, 0).
# The state now changes according to a special rule: Each cell changes depending on the
# state of the cell itself and its left and right neighbour. The cell changes to state 1
# whenever the values of c(<left nbr>, <itself>, <right nbr>) are
# (a) c(1, 0, 0), (b) c(0, 1, 1), (c) c(0, 1, 0), or (d) c(0, 0, 1).
# Each cell changes to state 0 (or remains 0) otherwise.
# The evolution of the vector above would be
# > c(0, 0, 0, 1, 0, 0, 0)
# > c(0, 0, 1, 1, 1, 0, 0)  # case (d) applies to element 3, case (b) to 4, case (a) to 5
# > c(0, 1, 1, 0, 0, 1, 0)  # case (d) applies to 2, (b) to 3, (a) to 6. Some cells switch back to 0.
# > c(1, 1, 0, 1, 1, 1, 1)  # case (d) to 1, (b) to 2, (a) to 4, (d) to 5, (c) to 6, (a) to 7
#
# Write a function that takes as input a vector of 0s and 1s and "evolves" the state for a given
# number of steps (the initial state counting as first step). The return value should be a
# *matrix* of states in each step, with the first row being the input vector.
# The upper example would look like this:
# > initial.state = c(0, 0, 0, 1, 0, 0, 0)
# > steps = 4
# --> returns the value as of
# rbind(c(0, 0, 0, 1, 0, 0, 0),
#       c(0, 0, 1, 1, 1, 0, 0),
#       c(0, 1, 1, 0, 0, 1, 0),
#       c(1, 1, 0, 1, 1, 1, 1))
#
# The "boundaries", i.e. cells to the left or right of the vector, are
# always set to 0, for example:
# > initial.state = 1
# > steps = 5
# --> returns a value stuck at 1, because case (c) applies every time:
# > rbind(1, 1, 1, 1, 1)
#
# Another example:
# > initial.state = c(0, 0, 1)
# > steps = 5
# -->
# rbind(c(0, 0, 1),
#       c(0, 1, 1),  # case (d) for position 2, case (c) for position 3
#       c(1, 1, 0),  # case (d) for position 1, case (b) for position 2
#       c(1, 0, 1),  # case (b) for position 1, case (a) for position 3
#       c(1, 0, 1))  # case (c) applies to both position 1 and 3
#
# This particular cellular automaton is called "Rule 30" <https://en.wikipedia.org/wiki/Rule_30>.
# You can generate some beautiful pictures when displaying the function results using `image()`.
# There are some unsolved mathematical problems about Rule 30, some of which Stephen Wolfram has set out
# USD 30,000 Prizes for: <https://writings.stephenwolfram.com/2019/10/announcing-the-rule-30-prizes/>.
# (Although if you can get one of these then this is probably the wrong course for you)
ex02CellularAutomaton <- function(initial.state, steps) {
  # your code here
  result <- matrix(0, nrow = steps, ncol = length(initial.state))
  result[1, ] <- initial.state
  for (st in seq_len(steps - 1)) {
    prevstep <- c(0, result[st, ], 0)  # pad with 0 so we can apply the rules to the edge positions
    for (position in seq_along(initial.state)) {
      prevwindow <- prevstep[position:(position + 2)]
      prevwindow.str <- paste(prevwindow, collapse = "")  # turn c(1, 0, 0) into "100" etc.
      result[st + 1, position] <- prevwindow.str %in% c("100", "011", "010", "001")
    }
  }
  result
}
