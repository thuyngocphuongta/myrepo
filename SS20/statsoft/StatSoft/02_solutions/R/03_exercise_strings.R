
# This exercise is concerned with string operations. Use the base R string operations
# (grep, gsub, paste0, substr, strsplit, sprintf, nchar, ...) to solve these.
# There are other R-libraries that also offer string operations, but they should not
# be used here.

# Palindromes are words or sentences that, when read backwards, give the same sentence,
# ignoring punctuation, spacing, or capitalization. Examples are
# > "Was it a car or a cat I saw?"
# > "Go hang a salami, I'm a lasagna hog"
# Write a function that returns TRUE if a given string is a palindrome, FALSE otherwise.
# The input value is a character vector with one element (written `character(1)` and what we
# call a "string"). You can rely on there only being latin letters, punctuation marks and
# spaces being present, and that the string contains at least one letter.
ex01Palindrome <- function(input) {
  # your code
  str <- strsplit(toupper(input), "")[[1]]
  str <- str[str %in% LETTERS]
  all(str == rev(str))
}

# Write a function that finds all URLs in a text. Input is a long string containing
# text, some parts of which are URLs.
#
# A URL is the sequence 'http://' or 'https://' (where the http/https-part indicates the PROTOCOL)
# if not immediately preceded by another letter ("abchttp://" does not start a url, but "abc http://" may),
# followed by the DOMAIN NAME (containing two or more "labels" separated by a dot,
# where each label consists of letters, numbers or minus-signs but does not start with a minus-sign;
# labels contain at least one character), followed by "/", followed by a PATH. We limit ourselves to
# PATHs that contain only latin letters, numbers, minus-signs and "/"s (slashes). All these rules
# are case-insensitive, so a URL may start with HTTP or hTTp.
#
# Given an input text, return a `data.frame` with the character columns "protocol", "domainname" and
# "path", containing the information about all the URLs found *in the order they were found*. (The same URL may
# occur multiple times and should then be listed in the data frame multiple times). Make sure you don't include
# things that are not URLs because they don't satisfy all the rules.
#
# Example input:
# > "https://www.google.com/ is probably the most popular url, sometimes used ashttp://www.google.com/ if one forgets
#    to use the more secure https:// protocol. Another URL that is popular is http://youtube.com which is owned
#    by the same company. On some computers, it is possible to find a website at http://localhost/ if the
#    computer runs a webserver. This exercise can be found at HTTPS://GITHUB.COM/PROGR-2020/02_STRUCTURED_PROGRAMMING."
# Output:
# > data.frame(protocol = c("https", "HTTPS"), domainname = c("www.google.com", "GITHUB.COM"),
#     path = c("", "PROGR-2020/02"), stringsAsFactors = FALSE)
# Example input:
# > "this text does not contain a url."
# Output:
# > data.frame(protocol = character(0), domainname = character(0),
#     path = character(0), stringsAsFactors = FALSE)
# Notes: many of the other occurrences of http.... do not count because they are either preceded directly by
# a letter, have no "/" following the domain name, or have a domain name with less than two labels.
# The path of the last URL is cut off early because we don't consider underscores as parts of the path.
#
# Be aware of the `stringsAsFactors = FALSE`, which you must use in most R-versions to prevent R from converting
# the data frame columns to factors.
#
# You should probably look into `regexpr`, `gregexpr`, `regexec` and similar to solve this problem.
ex02UrlFinder <- function(input) {
  # your code
  matches <- gregexpr(
    "(^|[^a-z])(?P<protocol>https?)://(?P<domainname>[a-z0-9][-a-z0-9]*(\\.[a-z0-9][-a-z0-9]*)+)/(?P<path>[-0-9a-z/]*)",
    input, perl = TRUE, ignore.case = TRUE)[[1]]

  if (matches[[1]] == -1) {
    return(data.frame(
        protocol = character(0), domainname = character(0), path = character(0),
        stringsAsFactors = FALSE
    ))
  }

  matchmat <- sapply(c("protocol", "domainname", "path"), function(groupname) {
    submatch <- attr(matches, "capture.start")[, groupname]
    attr(submatch, "match.length") <- attr(matches, "capture.length")[, groupname]
    regmatches(input, list(submatch))[[1]]
  }, simplify = FALSE)

  as.data.frame(matchmat, stringsAsFactors = FALSE)
}


# This function gets two arguments: `parent` (character vector length 1, i.e. string)
# and `children` (character vector of arbitrary length).
# Return a single string describing this family in human words.
# E.g. parent = "Eric", children = c("Bob", "Helga") --> `Eric has 2 children: "Bob" and "Helga".`
#      parent = "Herbert", children = "Klaus Dieter" --> `Herbert has 1 child: "Klaus Dieter".`
#      parent = "Hildegard", children = character(0) --> `Hildegard has no children.`
#      parent = "Y", children = c("A", "B", "C")     --> `Y has 3 children: "A", "B" and "C".`
# Watch out for punctuation (comma, quotation around children but not parent, period at the end),
# singular vs. plural-form of 'children' and the special case of 0 children.
ex03Children <- function(parent, children) {
  # your code
  if (!length(children)) {
    return(sprintf("%s has no children.", parent))
  }
  children <- sprintf('"%s"', children)
  child.denom <- if (length(children) == 1) "child" else "children"

  if (length(children) > 1) {
    childlist <- sprintf("%s and %s",
      paste(head(children, -1), collapse = ", "),
      tail(children, 1))
  } else {
    childlist <- children
  }

  sprintf("%s has %s %s: %s.",
    parent, length(children), child.denom, childlist
  )
}

# Now reverse the above: Given a string `X has Y children: ...`, extract the `children` argument
# from above. However, sometimes the number of children is wrong, in that case return "ERROR".
# E.g. `Eric has 2 children: "Bob" and "Helga".` --> c("Bob", "Helga")
#      `Herbert has 1 child: "Klaus Dieter".`    --> "Klaus Dieter"
#      `Hilde,gard has 2 children: "01 0" and ",,,".`    --> c("01 0", ",,,")
#      `Y has 10 children: "A", "B" and "C".`    --> "ERROR"
# (You can rely on the given sentence structure. There is always at least 1 child.
# The name of the parent does not contain spaces but may contain any other character.
# The name of children does not contain quotation marks but may contain any other character.)
ex04ChildrenInverse <- function(sentence) {
  # your code
  splt <- strsplit(sentence, " ")[[1]]
  expected <- as.numeric(splt[[3]])

  parentmatch <- regexpr("^[^ ]*", sentence)  # remove parent from string, which could contain quotation marks
  regmatches(sentence, parentmatch) <- ""

  childmatch <- gregexpr('"[^"]*"', sentence, perl = TRUE)
  children <- regmatches(sentence, childmatch)[[1]]
  children <- trimws(children, whitespace = '"')  # remove quotation marks (") from children's names

  if (length(children) != expected) "ERROR" else children
}


# The "Vignere" Cipher is a method of encrypting text that has once been widely used.
# It uses a `plaintext` and a `key` and generates encrypted text. This works by first
# repeating the `key` until one gets a string with the same length as the plaintext.
# Then each letter is converted to a number corresponding to its position in the alphabet.
# The letter values of the repeated key and the plaintext are added and converted back
# to letters (modulo the number of letters). Decryption can be done by subtracting,
# instead of adding, the key.
#
# We are working with the alphabet of the space character " " (value 0) and the 26 capital
# letters of the latin alphabet (contained in the R variable `LETTERS`, numbered 1..26.)
#
# Example:
# plaintext     = I LOVE MY PARENTS KYLIE MINOGUE AND KERMIT THE FROG
# key           = LEMON
# repeated key  = LEMONLEMONLEMONLEMONLEMONLEMONLEMONLEMONLEMONLEMONL
# plaintext converted to numbers:
#               = c( 9, 0, 12, 15, 22,  5, 0, 13, 25,  0, 16, 1, 18,  5, 14, 20, 19,  0, .....
# repeated key converted to numbers:
#               = c(12, 5, 13, 15, 14, 12, 5, 13, 15, 14, 12, 5, 13, 15, 14, 12,  5, 13, .....
# sum of these two
#               = c(21, 5, 25, 30, 36, 17, 5, 26, 40, 14, 28, 6, 31, 20, 28, 32, 24, 13, .....
# some of the values are larger than 26, so we have to wrap them back around (modulo 27 -- note
# how we have 27 letters: 26 in the alphabet plus the space!):
# sum of plaintext and key with modulo:
#               = c(21, 5, 25,  3,  9, 17, 5, 26, 13, 14,  1,  6, 4, 20,  1,  5, 24, 13, .....
# converted back to letters:
#               = UEYCIQEZMNAFDTAEXMZLXNRO USAVHQENBRLPRF UYMHVQESFBS
#
# A few more examples:
# plaintext: COME LETS EAT GRANDPA
# key:       ABC
# result:    DQPFBOFVVAGDUBJSCQERD
# plaintext: I LIKE COOKING MY FRIENDS AND MY FAMILY
# key:       " " (space)
# result:    I LIKE COOKING MY FRIENDS AND MY FAMILY
#            (no encryption because " " corresponds to 0, so values are not changed)
# Implement a function that performs Vignere encryption or decryption, taking one
# `plaintext` and one `key` parameter. You can assume both only consist of uppercase letters
# and the space character. If `decrypt` is TRUE, then decryption should be performed instead
# of encryption (subtraction instead of addition).
# You may find the `match()` function and the modulo operator %% useful.
#
# Be aware that this cipher is very insecure and you should not use it to actually hide information.
# You can read more about the cipher at Wikipedia: <https://en.wikipedia.org/wiki/Vigen%C3%A8re_cipher>
ex05VignereCipher <- function(plaintext, key, decrypt = FALSE) {
  # your code
  code <- c(" ", LETTERS)
  plaintext <- strsplit(plaintext, "")[[1]]
  key <- strsplit(key, "")[[1]]
  key <- rep(key, 1 + length(plaintext) / length(key))
  decryptmultiplier <- if (decrypt) -1 else 1
  numbers <- (match(plaintext, code) - 1) + (match(key, code)[seq_along(plaintext)] - 1) * decryptmultiplier
  numbers <- numbers %% 27
  paste(code[numbers + 1], collapse = "")
}
