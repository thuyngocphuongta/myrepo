### Aufgabe 2

# zur Generierung bivariater normalverteilter Zufallszahlen
#install.packages("MASS")
library(MASS)


#Definition der Kovarianzmatrix
S <- matrix(c(2,2,2,5), nrow=2, ncol=2)


# zunächst einfach: Generierung einer Wishart-verteilten Zufallsmatrix mit
# Parameter S und m=1


x <- mvrnorm(n = 1, mu = c(0,0), Sigma = S) # eine 2-dim normalverteile ZV
x
M <- x[1] %*% t(x[1])    # eine Wishart-verteile ZV mit m=1
M


#Funktion zur Berechnung von n Wishart-verteilten Zufallsmatrizen mit Parametern
#S und m
m <- 10 # Freiheitsgrade
n <- 100


rwishart <- function(n,S,m){

  mu1 <- rep(0,length(S[,1]))  #Definition von \mu=0
  #Dimension von Sigma liefert, dass eine 2-dimensionale
  #Multivariate Normalverteilung zugrunde liegt!

  #Simulation der normalverteilten Zufallsvektoren
  A <- mvrnorm(n = n*m, mu = mu1, Sigma = S) # simuliere 2-dim. normalvertielte Zufallszahlen mit Erwartungswert 0 und
  # Kovarianzmatrix Sigma
  # Ben?tigt werden n*m Zufallszahlen, da f?r jede Wishart-verteilte
  # Zufallszahl m normalverteilte Zufallszahlen ben?tigt werden!
  # dim(A)
  wlist <- list()  # Liste, weil die Zufallszahlen Matrizen sind (der Dimension = dim(S))
  #Und nun jeweils m xx' aufsummieren zu einer Wishart Zufallsvariable
  for(j in 1:n){
    W <- 0
    # f?r die erste wishart-verteilte Zufallszahl die ersten m normalverteilten Zufallszahlen
    # f?r die zweite wishart-verteilte Zufallszahl die zweiten m normalverteilten Zufallszahlen
    # ...
    # f?r die n-te wishart-verteilte Zufallszahl die letzten m normalverteilten Zufallszahlen
    for(i in ((j-1)*m+1):((j-1)*m+m)){
      W <- W + A[i,] %*% t(A[i,])
    }
    #Liste, die die Wishart-verteilten Zufallsvariablen enth?lt
    wlist[[j]] <- W
  }
  return(wlist)
}

# Ziehe 100 wishart-verteilte Zufallszahlen mit Kovarianzmatrix S und 10 Freiheitsgraden
wishartsample <- rwishart(n=100, S=S, m=10)
wishartsample[[1]]
wishartsample[[100]]

#Das ist eine Liste von W(S,m)-verteilten Matrizen.

#direkt aus Wishart-Verteilung
p <- rWishart(n = 100, df = 10, Sigma = S)
p


