## app.R ##
library(shiny)
library(shinydashboard)
library(DT)
library(plotly)
library(dplyr)
library(scales)
library(reshape)
library(shinyWidgets)

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
              # introductory texxt
              h1('Analysing public attitudes using Twitter data'),
              p("The data for this dashboard was collected on Tuesday 11th Oct using the Blattodea tool."),
              p("The tool is an wrapper for snscrape that pulls Tweets from a searched user and the tweets from a customizable number of people within their network. The tool also retrieves the tweet content, including some stats about the Tweet at the time of collection (e.g. nr of likes, retweets). The users that were initially searched for were @elonmusk and @ZelenskyyUa. The reasons behind this was that I wanted to try to answer the following:"),
              p("   1. Have most recent controversial tweets by the former had any impact that could be seen at the level of tweet metadata?"),
              p("   2. Has been any change in the tweet metadata of for the latter, especially after his response to @elonmusk's controversial twitter poll?"),
              p("   3. Can we use NLProc techniques to analyse the corpus of tweets and find any useful insights."),
              p("For the last point, @elonmusk is a good choice, since he has a tendency to tweet about a variety of divergent topics."),
              br(),
            
              # next row 
              fluidRow(
                box(plotlyOutput('before_and_after_mean_plot', height=300)), 
                box(plotOutput('before_and_after_boxplot', height=300))
              ),
              
              fluidRow(
                box(pickerInput('tweet_stats_variable', label='Which Twitter Stats for @elonmusk to show', 
                                choices = c('Number of likes', 
                                            'Number of retweets', 
                                            'Number of responses'
                                            ),
                                selected=c("Number of responses"),
                                options = list(`actions-box` = TRUE),
                                multiple = T
                                ))
                ),
              
              fluidRow(
                box(plotlyOutput('TS_tweet_stats', height=450, width="100%"), width=12)
              ),
              
              fluidRow(
                box(
                  # title='Accounts with most followers within dataset',
                  sliderInput('N_slider_followers', 'Most followers: Number of accounts', 0, 100, 20),
                  height = 100
                ) ,
                box(
                  # title='Accounts with most followers within dataset',
                  sliderInput('N_slider_posts', 'Most posts: Number of accounts', 0, 100, 20),
                  height = 100
                )
              ),
              
              fluidRow(
                box(plotlyOutput('followers_barplot')), 
                box(plotlyOutput('posts_barplot'))
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
               mainPanel(title='Clean_data', dataTableOutput('Clean_data')
               
              )
      )
    )
    
  
)
)
)




server <- function(input, output) {set.seed(122)
  
  
  ### DATA ETL AND WRANGLING HERE #########################################
  disp_df <- read.csv('../data/clean/clean_display_data.csv')

  #getting user summary stats
  df <- read.csv('../data/raw/user_attributes.csv')
  df <- distinct(df) 
  top_n <- 20
  print(names(df))
  df<- df %>%
    dplyr::rename(
      Handle = X,
      Followers =  X.followers,
      Posts = X.posts,
      Friends = X.friends,
      Verified = verified
    )
  print(names(disp_df))
  disp_df<- disp_df %>%
    dplyr::rename(
      Display_name = display_name,
      Number_likes= X.likes,
      Number_retweets = X.retweets,
      Number_responses = X.responses
    )
  
  df$Verified <- gsub('True', 'Verified', df$Verified)
  df$Verified <- gsub('False', 'Not verified', df$Verified)
  
  countfollow <- df[c('Handle', 'Followers','Posts', 'Verified')][order(df$Followers, decreasing=TRUE),]
  countpost <- df[c('Handle', 'Posts','Followers', 'Verified')][order(df$Posts, decreasing=TRUE),]

  #now another dataframe which handles number of likes before and after event
  # target_df <- subset(disp_df,  grepl(targets, extracted_twitter_handles))
  target_df <- subset(disp_df, Display_name=='ElonMusk')
  
  # now groupby and sum
  
  target_mean <- target_df %>% group_by(Before_or_after_controversy) %>%
    summarise('Number of likes' = mean(Number_likes),
              'Number of retweets' = mean(Number_retweets),
              'Number of responses' = mean(Number_responses)
    )
  target_mean <- as.data.frame(target_mean)
  
  target_mean <- target_mean %>% 
    dplyr::rename('Before or after controversial tweet' = Before_or_after_controversy)
  #melt it so it's easier to plot
  target_mean_melt <- reshape::melt(target_mean, id = c('Before or after controversial tweet'))
  target_mean_melt <- as.data.frame(target_mean_melt)
  target_mean_melt$`Before or after controversial tweet` <- factor(target_mean_melt$`Before or after controversial tweet`,
                                                                   levels = c('Before', 'After'))
  
                                                 
                                                                   
  ## dataframe for boxplot
  target_df <- target_df %>%
    rename( 'Number of likes' = 'Number_likes', 
            'Number of retweets' = 'Number_retweets', 
            'Number of responses' = 'Number_responses')
  
  
  target_df_melt<- reshape::melt(target_df[c('Number of likes', 
                                             'Number of retweets', 
                                             'Number of responses', 
                                             'Before_or_after_controversy') ], 
                                 id='Before_or_after_controversy')
  target_df_melt$Before_or_after_controversy<- factor(target_df_melt$Before_or_after_controversy, 
                                                      levels = c('Before', 'After'))
  
  target_df_melt <- target_df_melt %>% 
    dplyr::rename('Before or after controversial tweet' = Before_or_after_controversy)
  
  
  
  time_target_df <- target_df
  time_target_df$DT <- as.POSIXct(time_target_df$datetime)
  time_target_df <- time_target_df %>%
    select('DT', 'Display_name', 
           'clean_tweet_text', 
           'Number of likes', 
           'Number of retweets', 
           'Number of responses')
  
  # time_target_df <- time_target_df %>%
  #   rename( 'Number of likes' = 'Number_likes', 
  #           'Number of retweets' = 'Number_retweets', 
  #           'Number of responses' = 'Number_responses')

    
  time_target_melt<- reshape::melt(time_target_df, id.vars=c('DT', 'Display_name', 'clean_tweet_text'))
  time_target_melt
  
  
  cutoff_date <- as.POSIXct('2022-10-03 16:15:43+0000')
  # wrapping teh text
  time_target_melt$clean_tweet_text = lapply(time_target_melt$clean_tweet_text, function(x){stringr::str_wrap(x, 15)})
  time_target_melt$clean_tweet_text = lapply(time_target_melt$clean_tweet_text, function(x){gsub('\n', '<br>', x)})
  
  time_target_melt$clean_tweet_text
  
  
  ########################################################################
  
  
  ### PLOTS BEING GENERATED HERE #########################################
  
  
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
      layout(yaxis = list(categoryorder = "total ascending",title=NA),
             xaxis=list(title=NA)
                          ) %>%
      layout( categoryorder = "total ascending", 
              yaxis=list(title=NA), 
              xaxis = list(title=NA))  %>%
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
      layout(yaxis = list(categoryorder = "total ascending", list(title=NA) ), 
             xaxis = list(title=NA)
                          ) %>%
      config(modeBarButtons = list(list("toImage")), displaylogo = FALSE, toImageButtonOptions = list(filename = "plotOutput.png"))
  })
  
  output$before_and_after_mean_plot <- renderPlotly({
    
      ggplotly(
        ggplot(data= target_mean_melt)+ 
          geom_bar(aes(x=variable, y= value, 
                       fill= `Before or after controversial tweet`),
                   stat='identity', position='dodge')+
          labs(title='Before and after @elonmusk\'s poll', 
               subtitle = 'How have the average stats for Elon\'s Tweets changed\nafter his poll on the war?',
               y = '', 
               x='')+
          # coord_flip()+
          theme_minimal()+
          theme(legend.title=element_blank(),
                axis.text.x=element_text(angle=45, vjust=0, hjust=1)
                )

            ) %>%
      layout(legend=list(title=list(text='')), 
             title= list(subtitle = list(text = 'How have the average stats for Elon\'s Tweets changed\nafter his poll on the war?'))) %>%
      config(modeBarButtons = list(list("toImage")), displaylogo = FALSE, toImageButtonOptions = list(filename = "plotOutput.png"))
    
  }) 
  


  # boxplot being generated
  output$before_and_after_boxplot<- renderPlot({
    

    ggplot(
      target_df_melt
    )+
      geom_jitter(aes(y=value, 
                      x=variable, 
                      colour=`Before or after controversial tweet`
      ),
      alpha=0.25, 
      position =position_jitterdodge(
        dodge.width=0.95,
        jitter.width=0.5,
        jitter.height=0
      )
      )+
      geom_boxplot(
        aes(
          y= value,
          x=variable,
          fill= `Before or after controversial tweet`
        ),
        outlier.shape = NA,
        outlier.alpha = 0.01,
        outlier.colour = 'white',
        outlier.fill='white',
        position= position_dodge(width =0.9))+
      
      labs(title='Before and after the poll (excluding extreme outliers)', 
           subtitle = 'How have the average stats for Elon\'s Tweets\nchanged after his poll on the war?', 
           x = '', 
           y='')+
      scale_y_continuous(limits = quantile(target_df_melt$value, c(0.0, 0.9)))+
      theme_minimal()+
      coord_flip()+
      theme(legend.title=element_blank(),
            text = element_text(family = 'Open Sans')
            # axis.text.x=element_text(angle=45, vjust=1.5, hjust=1)
            )
    
  })
  
  
  output$TS_tweet_stats <- renderPlotly({
    plot_ly(
      data=subset(time_target_melt, variable==input$tweet_stats_variable), 
      x = ~DT, 
      y = ~value,
      color=~variable,
      text=~clean_tweet_text,
      hovertemplate = paste('<br><b>Date</b>:%{x}',
                            # '<br><b>%{color}</b>: '
                            '<br>%{y}',
                            '<br><b>Text</b>:%{text}',
      title="How have @elonmusk\'s tweets fared in this period?"
      ),
      type = 'scatter', mode='markers')  %>% 
      layout(yaxis=list(title=NA), 
             xaxis = list(title=NA), 
             title = list(title="How have @elonmusk\'s tweets fared in this period?", 
                        xanchor = "right") )
  })
  
  
  
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

    
