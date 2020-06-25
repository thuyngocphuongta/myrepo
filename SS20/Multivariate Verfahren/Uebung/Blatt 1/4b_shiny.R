library(shiny)
library(ggplot2)

wdi <- readRDS("wdi_daten.rds")


shinyApp(
  ui = shinyUI(fluidPage(
    
    titlePanel("Shiny - Beispiel"),
    
    sidebarLayout(
      sidebarPanel(
        selectInput("jahr", label = "Wählen Sie das Jahr aus:",
                    choices = 2010:2013, selected = 2013),
        sliderInput("bw.x", label = "Bandweite in x-Richtung:",
                    min = 10, max = 100, step = 10, value = 50),
        sliderInput("bw.y", label = "Bandweite in y-Richtung:",
                    min = 10, max = 100, step = 10, value = 50)
      ),
      
      mainPanel(
        plotOutput("scatterplot")
      )
    )
  )),
  server = shinyServer(function(input, output) {
    
    output$scatterplot <- renderPlot({
      ggplot(wdi[wdi$Jahr == input$jahr,], aes(x = ZS, y = KSL)) +
        geom_point() + xlim(c(0, 100)) +
        geom_density2d(h = c(input$bw.x, input$bw.y))
    })
  })
)
