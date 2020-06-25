### Aufgabe 1
# n=3 Zeilen und p=2 Spalten
X <- matrix(nrow=3, ncol=2)
X[,1] <- c(3,1,2) # Spalte 1
X[,2] <- c(2,5,5) # Spalte 2
X

### a
# Spaltenmittelwerte
mean(X[,1])
mean(X[,2])


## schnell
# MARGIN=2 => wende Funktion mean() auf die Spalten von X an!
x_quer <- apply(X , 2 , mean)
x_quer

#schneller vor allem bei größeren Matrizen
w <- matrix(c(1:64), nrow = 8)
w_quer <- apply(w, 2, mean)
w_quer

### b (iii)
## mit Formel für erwartungstreuen Sch?tzer aus Folie 9
n <- nrow(X)
sum_xi_txi <- X[1,] %*% t(X[1,]) + X[2,] %*% t(X[2,]) + X[3,] %*% t(X[3,])
S <- 1/(n-1) * (sum_xi_txi - n*x_quer %*% t(x_quer))
S

## mit Zentrierungsmatrix H aus Folie 10
H <- diag(n)-(1/n)*matrix(1,nrow=n,ncol=n)
H
S <- 1/(n-1) * t(X) %*% H %*% X
S

## schnell
cov(X)
