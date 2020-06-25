################################################################################
# Shiny-App für Aufgabe 4 des ersten Tutoriumsblattes für multivariate Verfahren
#
# Die Shiny-App visualisiert einen Scatterplot der Variablen holiday und
# earningHour aus dem Datensatz ses aus dem laeken-Paket. Aus dieser Grafik
# können Beobachtungen mit einer Drag-Box selektiert werden. Von den
# ausgewählten Beobachtungen werden Kennzahlen ausgegeben.
################################################################################

# 1. Laden notwendiger Pakete und der Daten
library(plotly)
library(shiny)
library("laeken")
data("ses")

ses_small <- ses[sample(nrow(ses), 1000), ]

# 2. Definition des User-Interfaces:
#     Die Ausgabe soll zwei Teile untereinander anzeigen:
#       zuerst den Scatterplot
#       anschließend die Kennzahlen
ui <- fluidPage(
  plotlyOutput("plot"),
  verbatimTextOutput("return")
)

# 3. Definition des Servers:
server <- function(input, output, session) {

#     Erstellung des Scatterplots mit Drag-box-Auswahl:
  output$plot <- renderPlotly({
    p <- ggplot(aes(x=holiday, y=earningsHour),data=ses_small)+ylim(c(0,130)) +
      geom_point()
    ggplotly(p) %>% layout(dragmode = "select")
  })

#     Berechnung des Outputs:
  output$return <- renderPrint({
    selection <- summary(event_data("plotly_selected"))
    selection
  })
}

# 4. Zusammenführung zu einer App:
shinyApp(ui, server)
