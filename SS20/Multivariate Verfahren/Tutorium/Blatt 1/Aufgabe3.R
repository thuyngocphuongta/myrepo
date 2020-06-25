################################################################################
# Beispiellösung für Aufgabe 3 des ersten Tutoriumsblattes für multivariate Verfahren
################################################################################

# Laden der Daten
#install.packages(ggplot2)
#install.packages(laeken)
library(laeken)
library(ggplot2)
data(ses)
data("eusilc")
?ses

################################################################################
# Einführung in ggplot2:

# Grammatik von Grafiken:
# 1. Definition der Grafik:
a <- ggplot()
# 2. Definition von Ebenen (layer) bestehend aus
#     a) dem Plottyp, z.B. geom_point() für Scatterplots
#     b) den Daten, übergeben wie bei gewöhnlichen R-Grafiken
#     c) aesthetic mappings, d.h. welche Variablen werden geplottet, 
#         z.B. x=earningsMonth als X-Variable und y=earningsHour als Y-Variable
#     d) Transformationen der Daten stat, z.B. logarithmieren
#     e) Anordnen der Grafikelemente
a+geom_point(aes(x=holiday, y=earningsHour),data=ses)
# Die Dimensionen einer Ebene können auch in ggplot() übergeben werden und
# werden dann als Defaultwerte für alle Ebenen betrachtet.
ggplot(data=ses, aes(x=holiday, y=earningsHour))+geom_point()+geom_smooth()
# Es kann sich aber auch jede Dimension jeder Ebene unterscheiden, was nicht immer Sinn ergibt:
ggplot()+geom_point(aes(x=holiday, y=earningsHour),data=ses)+geom_smooth(data=eusilc, aes(x=hy145n, y=age))


## Aufgabe 3.1
## Verschiedene Scatterplots
p <- ggplot(aes(x=holiday, y=earningsHour),data=ses)+ylim(c(0,130))
p + geom_point()
#Problem? Overplotting
p+geom_point(alpha=0.1) #Transparenz
p+geom_hex(bins=50) #Hexbin-Scatterplot
#Gibt es einen Trend? smooth ergänzen
p+geom_point(alpha=0.1)+geom_smooth()

## Aufgabe 3.2
## Scatterplot mit Randverteilungen:
library(gridExtra)
main <- p + geom_point()
top <- ggplot(ses, aes(x = holiday)) + geom_histogram()
right <- ggplot(ses, aes(x = "", y = earningsHour)) + 
  geom_boxplot() + ylim(c(0,130))
grid.arrange(top,
             ggplot() + theme_minimal(),
             main,
             right,
             nrow = 2, ncol = 2)

## Aufgabe 3.3
## Aufnahme von zusätzlichen Variablen:
p+geom_point(alpha=0.5, aes(col=sex))+geom_smooth()
# Problem: Durch Overplotting vermischen sich die Farben
p+geom_point(alpha=0.5, aes(col=location))
# Hier wird das noch deutlicher
p+geom_point(alpha=0.5, aes(col=education,shape=sex))
# Jetzt erkennt man nichts mehr
# Fazit: Drittvariablen durch Farbe, Grö?Ye der Punkte oder unterschiedliche
#        Formen mit aufzunehmen geht nur bei "kleinen" Datensätzen.

## Aufgabe 3.4
# Nur Vollzeitarbeiter, die mehr als 50 Wochen gearbeitet haben:
ses_full <- ses[ses$fullPart=="FT"&ses$weeks>50,]
#install.packages("aplpack")
require("aplpack")


# Bivariater Boxplot:
bagplot(ses_full$holiday, ses_full$earningsHour)
# Im inneren Kreis befinden sich 50% der Daten. Im Aeusseren befinden sich 95%.
# Die Striche zeigen die Distanz von Ausreissern zu den "inneren 95%" an.

## Aufgabe 3.5
# Erstellung eines Parallel-(Koordinaten-)plots
#install.packages("GGally")
ses2 <- ses[1:100,]
require(GGally)
?ggparcoord()
ggparcoord(data = ses2, columns = 15:20, scale="std", groupColumn = 7, order = "anyClass",
           showPoints = TRUE, title = "Parallel Coordinate Plot for ses Data",
           alphaLines = 0.5)
# Nachteil dieser Grafik: Die Informationen jeder Variable werden nur sehr
# vereinfacht dargestellt. Die Anordnung der Achsen kann den subjektiven Eindruck
# stark beeinflussen.

# Vorteil dieser Grafik: Zusammenhänge zwischen vielen Variablen können schön
# visualisiert werden. (Das ist nicht immer der Fall, z.B. nicht, wenn die
# Zusammenhänge sich zwischen mehreren Gruppen unterscheiden.)


## Aufgabe 4
runApp("shinyapp4.R")

