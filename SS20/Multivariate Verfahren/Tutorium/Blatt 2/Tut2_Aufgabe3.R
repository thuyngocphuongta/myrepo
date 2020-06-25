#Aufgabe 3

# Im Datensatz decathlon im R-Paket FactoMineR befinden sich die Leistungsdaten von 41
# Zehnkaempfern. Fur diese Aufgabe sollen die Zeiten des 100-Meter-Laufs (M100) 
# und des 400-Meter-Laufs (M400) betrachtet werden. Berechnen Sie alle 
# notwendigen Informationen fur diese Aufgabe aus diesen Daten mit R


#install.packages("FactoMineR")
library(FactoMineR)

#Daten einlesen
data(decathlon)
#oder
decathlon <- read.table(file="http://www.sthda.com/upload/decathlon.txt",
                        header=T, row.names=1, sep="\t")
View(decathlon)


# Testen Sie zum Signifikanzniveau von 5% die Hypothese H0: mu = mu0
# T = n * (x_quer - mu0)^T * Sigma^-1 * (x_quer - mu0)

#Mittelwerte berechnen
m100 <- decathlon[,"X100m"]
m400 <- decathlon[,"X400m"]
x1_quer <- mean(m100)
x2_quer <- mean(m400)

#Mittelwertsvektor
x_quer <- c(x1_quer, x2_quer)
x_quer


#oder wieder mit Apply

m100_m400 <- matrix(c(decathlon[,"X100m"], decathlon[,"X400m"]), ncol=2)
m100_m400
x_quer <- apply(m100_m400 , 2, mean)
x_quer

#Nullhypothese
mu0 <- c(11,50)
mu0

#Kovarianzmatrix
sigma <- matrix(c(2,0,0, 4), nrow=2)
sigma

#Teststatistik berechnen

n <- nrow(m100_m400)
n

T_2 <- n * t(x_quer - mu0) %*% solve(sigma) %*% (x_quer - mu0)
T_2

chi2_2 = qchisq(1-0.05, 2)
chi2_2
# Lehnt man ab?
T_2 > chi2_2

# Testentscheidung:
# T = 1.5088 < 5,99= Chi^2(2)
# H_0 kann zu einem 5% Signifikanzniveau nicht abgelehnt werden
# Die mittleren Ergebnisse unterscheiden sich nicht signifkant von c(11,50)

### Aufgabe 3b

# Schätzung Kovarianzmatrix aus Daten
S <- matrix(c(var(m100), cov(m100, m400),
              cov(m100, m400), var(m400)), nrow = 2)
S
# oder schneller:
S <- cov(m100_m400)
S
# Teststatistik berechnen

p <- 2
T2 <- ((n-p)*n)/(p*(n-1)) * t(x_quer - mu0) %*% solve(S) %*% (x_quer - mu0)
T2

# Testentscheidung
F_2_39 <- qf(1-0.05,2,39)
# Kann man ablehnen?
T2 > F_2_39
# T2 = 2,9632 < 3,23 = F_{2;39}
# H_O kann zu einem Signifikanzniveau von 5% nicht verworfen werdeen



#### Aufgabe 3c)
# verwende Test im Zwei-Stichprobenfall- unabhängige Stichproben

# Einteilung der Daten in zwei Gruppen
comp1 <- decathlon[decathlon[,"Competition"] == 1, ]
comp2 <- decathlon[decathlon[,"Competition"] == 2, ]



#Mittelwerte berechnen
mComp1_100 <- comp1[,"X100m"]
mComp1_400 <- comp1[,"X400m"]

mComp2_100 <- comp2[,"X100m"]
mComp2_400 <- comp2[,"X400m"]

x1_quer_100 <- mean(mComp1_100)
x1_quer_400 <- mean(mComp1_400)

x2_quer_100 <- mean(mComp2_100)
x2_quer_400 <- mean(mComp2_400)

#Mittelwertsvektor
mittel_1 <- c(x1_quer_100, x1_quer_400)
mittel_1
mittel_2 <- c(x2_quer_100, x2_quer_400)


# oder schneller:
mittel_1 <- apply(comp1[,c("X100m", "X400m")], 2, mean)
mittel_1
mittel_2 <- apply(comp2[,c("X100m", "X400m")], 2, mean)
mittel_2
#Kovarianzmatrix (wie in a)
S1 = matrix(c(var(mComp1_100), cov(mComp1_100, mComp1_400), 
              cov(mComp1_100, mComp1_400), var(mComp1_400)), nrow = 2)

S2 = matrix(c(var(mComp2_100), cov(mComp2_100, mComp2_400), 
              cov(mComp2_100, mComp2_400), var(mComp2_400)), nrow = 2)

n1 <- nrow(comp1)
n2 <- nrow(comp2)
S_p = 1 / (n1 + n2  - 2) * ( (n1-1) * S1 + (n2-1) * S2)

# Teststatistik berechnen
T_quad <- (n1 * n2) / (n1 + n2) * t(mittel_1 - mittel_2) %*% solve(S_p) %*% (mittel_1 - mittel_2)
T_quad_final <- (n1 + n2 - 2 - 1)/((n1 + n2 -2) * 2) * T_quad
T_quad_final

# Verteilung
F_2_38 <- qf(1-0.05,2, n1 + n2 - 2 - 1)
F_2_38

# Testentscheidung
# Kann man ablehnen?
T_quad_final > F_2_38
# Teststatistik = 7.81 > 3.24= F_{2;38}
# H_0 kann zu einem 5% Signifikanzniveau abgelehnt werden
# Die mittleren Ergebnisse beider Gruppen unterscheiden sich signifkant.

