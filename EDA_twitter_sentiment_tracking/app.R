## app.R ##
library(shiny)
library(shinydashboard)
library(DT)

ui <- dashboardPage(
  # def theme
  skin = 'blue',
  
  
  dashboardHeader(title = "Visualizing Twitter sentiment over time", titleWidth = 350),
  
  ## Sidebar content
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dashboard", tabName = "dashboard", icon = icon("dashboard")),
      menuItem("Clustering", tabName = "clustering", icon = icon("asterisk")),
      menuItem("Data", tabName = "data", icon = icon("th"))
    )
    
    ),
  
  ## Body content
  dashboardBody(
    tabItems(
      # First tab content
      tabItem(tabName = "dashboard",
              fluidRow(
                box(plotOutput("plot1", height = 250)),
                
                box(
                  title = "Controls",
                  sliderInput("slider", "Number of observations:", 1, 100, 50)
                )
              )
      ),
      
      
      # Second tab content
      tabItem(tabname = 'clustering',
              h2('Cluster Analysis'),


              fluidPage(
                mainPanel(title='Cluster_analysis'
                          # , plotlyOutput('graph')
                          )
              )
      ),


      
      # Third tab content
      tabItem(tabName = "data",
              h2("The Cleaned Twitter Data"),
              
              fluidPage(
               mainPanel(title='Clean_data', dataTableOutput('Clean_data'),
               
              )
      )
    )
    
  
)
)
)




server <- function(input, output) {set.seed(122)
  
  # primary tab with plots
  histdata <- rnorm(500)
  
  output$plot1 <- renderPlot({
    data <- histdata[seq_len(input$slider)]
    hist(data)
  })
  

  
  # cluster tab (nr 2) now being generated
  
  
  # etl tab of the clean data
  clean_dataframe <- read.csv('../data/clean/clean_display_data.csv')
  output$Clean_data <- renderDataTable({
    clean_dataframe
  }, 
  filter='top',
  rownames=FALSE,
  options = list(scrollX = TRUE)
  )
  
  }

shinyApp(ui, server)

    
