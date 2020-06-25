################################################################################
# Beispiellösung für Aufgabe 3 des ersten Tutoriumsblattes für multivariate Verfahren
################################################################################

# Laden der Daten
#install.packages(laeken)
library("laeken")
library("ggplot2")
data(ses)
data(eusilc)
?eusilc

################################################################################
# Einführung in ggplot2:

# Grammatik von Grafiken:
# 1. Definition der Grafik:
a <- ggplot()
# 2. Definition von Ebenen (layer) bestehend aus
#     a) dem Plottyp, z.B. geom_point() für Scatterplots
#     b) den Daten, übergeben wie bei gewöhnlichen R-Grafiken
#     c) aesthetic mappings, d.h. welche Variablen werden geplottet,
#         z.B. x=holiday als X-Variable und y=earningsHour als Y-Variable
#     d) Transformationen der Daten stat, z.B. logarithmieren
#     e) Anordnen der Grafikelemente
a+geom_point(aes(x=holiday, y=earningsHour),data=ses)
# Die Dimensionen einer Ebene können auch in ggplot() übergeben werden und
# werden dann als Defaultwerte für alle Ebenen betrachtet.
ggplot(data=ses, aes(x=holiday, y=earningsHour))+geom_point()+geom_smooth()
# Es kann sich aber auch jede Dimension jeder Ebene unterscheiden, was nicht immer Sinn ergibt:
ggplot()+geom_point(aes(x=holiday, y=earningsHour),data=ses)+geom_smooth(data=eusilc, aes(x=hy145n, y=age))


