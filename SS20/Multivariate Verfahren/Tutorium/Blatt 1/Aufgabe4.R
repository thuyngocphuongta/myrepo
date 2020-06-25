################################################################################
# Shiny-App für Aufgabe 4 des ersten Tutoriumsblattes für multivariate Verfahren
#
# Die Shiny-App visualisiert einen Scatterplot der Variablen holiday und
# earningHour aus dem Datensatz ses aus dem laeken-Paket. Aus dieser Grafik
# können Beobachtungen mit einer Drag-Box selektiert werden. Zus?tzlich kann
# eine beliebige Variable des Datensatz ausgew?hlt werden. Die App erstellt dann
# eine geeignete Grafik um die Verteilung der Variable zwischen den ausgew?hlten
# und den nicht ausgew?hlten Daten zu vergleichen.
################################################################################

# 1. Laden notwendiger Pakete und der Daten
library(plotly)
library(shiny)
library("laeken")
data("ses")
# eventuell Datensatz verkürzen
ses <- ses[1:1000,]

# 2. Definition des User-Interfaces:
ui <- fluidPage(
  
  #     Titel der App
  titlePanel("Holiday vs. earningHours aus dem ses-Datensatz"),
  
  #     Definition des Layouts: 
  sidebarLayout(
    
    #     Auswahlfeld an der Seite für die zu untersuchende Variable: 
    sidebarPanel(
      selectInput("var", label = "Zu untersuchende Variable:",
                  choices = names(ses), selected = "sex")
    ),
    
    #     Die Ausgabe soll zwei Teile untereinander anzeigen:
    #       zuerst den Scatterplot
    #       anschlie?Yend eine Grafik  
    mainPanel(
      plotlyOutput("plot"),
      plotOutput("return")
    )
  )
)

# 3. Definition des Servers:
server <- function(input, output, session) {
  
  #     Erstellung des Scatterplots mit Drag-box-Auswahl:
  output$plot <- renderPlotly({
    p <- ggplot(aes(x=holiday, y=earningsHour),data=ses)+
      ylim(c(0,130)) + 
      geom_point()
    
    ggplotly(p) %>% layout(dragmode = "select")
  })
  
  #     Erstellung der angeforderten Grafik:  
  output$return <- renderPlot({
    #       Erstellung der zu visualisierenden Daten:
    selection <- event_data("plotly_selected")
    d <- ses
    d$selected <- "ausgew?hlt"
    d$selected[d$holiday < min(selection$x)] <- "nicht ausgew?hlt" 
    d$selected[d$holiday > max(selection$x)] <- "nicht ausgew?hlt" 
    d$selected[d$earningsHour < min(selection$y)] <- "nicht ausgew?hlt" 
    d$selected[d$earningsHour > max(selection$y)] <- "nicht ausgew?hlt" 
    s <- d$selected
    v <- d[,input$var]
    d <- as.data.frame(cbind(s,v))
    
    #       Erstellung eines Balkendiagramms für Faktorvariablen:    
    if(is.factor(v)){
      r <- ggplot(data=d, aes(x=factor(s),fill=factor(v)))+
        xlab("Auswahl")+geom_bar(position="fill")+ 
        guides(fill=guide_legend(title=input$var))
    }
    
    #       Erstellung eines Boxplots für numerische Variablen:
    else{
      r <- ggplot(data=d, aes(factor(s),as.numeric(v)))+
        xlab("Auswahl")+geom_boxplot()+ 
        guides(fill=guide_legend(title=input$var))
    }
    r
  })
}

# 3. Zusammenführung zu einer App:
shinyApp(ui, server)   

