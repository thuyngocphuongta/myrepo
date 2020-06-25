### Übungsblatt 3

## Aufgabe 3

# b)

# Installation und Laden von benötigten Paketen:Ö
list_of_packages <- c("manipulate", "mvtnorm")
new_packages <- list_of_packages[!(list_of_packages %in%
                                     installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)
sapply(list_of_packages, require, character.only = TRUE)


# Funktion zum Erstellen eines Perspektivplots bezueglich einer bivariaten
# Normalverteilung:
nv2.3d <- function(mu1, mu2, sigma1, sigma2, rho, phi, theta){
  
  #Definition von Erwartungswert und Kovarianzmatrix
  mu <- c(mu1, mu2)
  Sigma <- matrix(c(sigma1^2, rep(rho * sigma1 * sigma2, 2), sigma2^2),
                  ncol = 2, nrow = 2)
  
  # Definition eines Grids von Werten zur Berechnung der Dichte:
  limits <- qnorm(0.975) * max(sigma1,sigma2) * c(-1, 1)
  steps <- seq(limits[1], limits[2], length.out = 50)
  x1 <- steps + mu1
  x2 <- steps + mu2
  grid <- expand.grid(x1,x2)
  
  # Berechnung der Dichtewerte:
  fvals <- matrix(dmvnorm(grid, mean = mu, sigma = Sigma), ncol = length(x1),
                  nrow = length(x2))
  
  # Drei-dimensionaler Plot der Dichte:
  persp(x = x1,y = x2,z = fvals, ticktype = "detailed", xlab = "\nx_1",
        ylab = "\nx_2", zlab = "\n\nf(x_1,x_2)", phi = phi, theta = theta)
  title(main = paste("mu1=",as.character(mu1), ", mu2=",as.character(mu2),
                     ", sigma1=",as.character(sigma1),
                     ", sigma2=", as.character(sigma2), ", rho=",
                     as.character(rho), sep = ""))
  
}

# Funktion zum Erstellen eines Contourplots bezueglich einer bivariaten
# Normalverteilung:
nv2.contour <- function(mu1, mu2, sigma1, sigma2, rho){
  
  #Definitition von Erwartungswert und Kovarianzmatrix
  mu <- c(mu1, mu2)
  Sigma <- matrix(c(sigma1^2, rep(rho * sigma1 * sigma2, 2), sigma2^2),
                  ncol = 2, nrow = 2)
  
  # Definition eines Grids von Werten zur Berechnung der Dichte:
  limits <- qnorm(0.975) * max(sigma1, sigma2) * c(-1, 1)
  steps <- seq(limits[1], limits[2], length.out = 50)
  x1 <- steps + mu1
  x2 <- steps + mu2
  grid <- expand.grid(x1,x2)
  
  # Berechnung der Dichtewerte:
  fvals <- matrix(dmvnorm(grid,mean=mu,sigma=Sigma),ncol=length(x1),nrow=length(x2))
  
  # Plot der Hoehenlinien:
  contour(x = x1, y = x2, z = fvals, xlab = "x_1", ylab = "x_2")
  title(main = paste("mu1=", as.character(mu1), ", mu2=", as.character(mu2),
                     ", sigma1=", as.character(sigma1), ", sigma2=",
                     as.character(sigma2), ", rho=", as.character(rho),
                     sep = ""))
  
}


#: Erstellen der interaktiven Graphiken:
doubleplot <- function(mu1, mu2, sigma1, sigma2, rho, phi, theta) {
  nv2.3d(mu1 = mu1, mu2 = mu2, sigma1 = sigma1, sigma2 = sigma2, rho = rho,
         phi = phi, theta = theta)
  nv2.contour(mu1 = mu1, mu2 = mu2, sigma1 = sigma1, sigma2 = sigma2,
              rho = rho)
}
par(mfrow = c(1, 2))
manipulate(doubleplot(mu1 = 0,mu2 = 0, sigma1, sigma2, rho, phi, theta),
           sigma1 = slider(1, 10, step = 1), sigma2 = slider(1, 10, step = 1),
           rho = slider(-0.9, 0.9, step = 0.1, initial = 0),
           phi = slider(0, 90, step = 9, initial = 45),
           theta = slider(0, 90, step = 9, initial = 45))
