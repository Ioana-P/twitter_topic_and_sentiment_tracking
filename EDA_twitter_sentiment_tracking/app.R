## app.R ##
library(shiny)
library(shinydashboard)
library(DT)
library(plotly)
library(dplyr)

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
              
              # EXAMPLE
              # fluidRow(
              #   box(plotOutput("plot1", height = 250)),
              #   
              #   box(
              #     title = "Controls",
              #     sliderInput("slider", "Number of observations:", 1, 100, 50)
              #   )
              # ),
              
              # next row of plot
              
              fluidRow(
                box(plotlyOutput('followers_barplot', height=400)),
                box(
                  # title='Accounts with most followers within dataset',
                  sliderInput('N_slider_followers', 'Most followers: Number of accounts', 0, 100, 20),
                  height = 100
                ), 
              ),
              
              
              #next row of plot
              fluidRow(
                box(plotlyOutput('posts_barplot', height=400)),
                box(
                  # title='Accounts with most followers within dataset',
                  sliderInput('N_slider_posts', 'Most posts: Number of accounts', 0, 100, 20),
                  height = 100
                )
              )
      ),
      
      
      # Second tab content
      tabItem(tabname = 'clustering',
              h2('Cluster Analysis'),


              fluidPage(
                mainPanel(title='Cluster_analysis',
                          plotlyOutput('graph')
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
  
  disp_df <- read.csv('../data/clean/clean_display_data.csv')

  #getting user summary stats
  df <- read.csv('../data/raw/user_attributes.csv')
  df <- distinct(df) 
  top_n <- 20
  
  df<- df %>%
    rename(
      Handle = X,
      Followers =  X.followers,
      Posts = X.posts,
      Friends = X.friends,
      Verified = verified
    )
  

  countfollow <- df[c('Handle', 'Followers','Posts', 'Verified')][order(df$Followers, decreasing=TRUE),]
  countpost <- df[c('Handle', 'Posts','Followers', 'Verified')][order(df$Posts, decreasing=TRUE),]

  
  # # primary tab with plots
  # histdata <- rnorm(500)
  # 
  # # examplere here
  # output$plot1 <- renderPlot({
  #   data <- histdata[seq_len(input$slider)]
  #   hist(data)
  # })
  
  
  #followers plot
  output$followers_barplot <- renderPlotly({
    plot_ly(data = head(countfollow, input$N_slider_followers),
            y=~Handle,
            x=~Followers,
            color=~Verified,
            text = ~Posts,
            type='bar',
            hovertemplate = paste('br>Handle</br>: %{y}', 
                                  '<br><b># posts</b>:%{text}<br>',
                                  '<br><b># followers</b>:%{x}<br>'
                                  )
    )%>% 
      layout(yaxis = list(categoryorder = "total ascending",
                          legend=list(title=list(text='<b> Account verified? </b>'))
                          )) %>%
    config(modeBarButtons = list(list("toImage")), displaylogo = FALSE, toImageButtonOptions = list(filename = "plotOutput.png"))
  })
  
  
  output$posts_barplot <- renderPlotly({
    plot_ly(data = head(countpost, input$N_slider_posts),
            y=~Handle,
            x=~Posts,
            color=~Verified,
            type='bar',
            text = ~Followers,
            hovertemplate = paste('<br>Handle</br>: %{y}', 
                                  '<br><b># posts</b>:%{x}<br>',
                                  '<br><b># followers</b>:%{text}<br>'
            )
    )%>% 
      layout(yaxis = list(categoryorder = "total ascending", 
                          legend=list(title=list(text='<b> Account verified? </b>'))
                          )) %>%
      config(modeBarButtons = list(list("toImage")), displaylogo = FALSE, toImageButtonOptions = list(filename = "plotOutput.png"))
  })

  
  # cluster tab (nr 2) now being generated
  
  
  # etl tab of the clean data
  
  output$Clean_data <- renderDataTable({
    disp_df
  }, 
  filter='top',
  rownames=FALSE,
  options = list(scrollX = TRUE)
  )
  
  }

shinyApp(ui, server)

    
