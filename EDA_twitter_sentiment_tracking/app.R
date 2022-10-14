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
      menuItem("Topics", tabName = "topics", icon=icon('asterisk')),
      menuItem("Data", tabName = "data", icon = icon("th"))
      )
    
    ),
  
  ## Body content
  dashboardBody(
    tabItems(
      # First tab content
      tabItem(tabName = "dashboard",
              # introductory text
              h1('Analysing public attitudes using Twitter data'),
              p("The data for this dashboard was collected on Tuesday 11th Oct using the ", tags$a(href="https://github.com/digital-scrappy/network-analysis-hackathon", "Blattodea tool.")),
              p("The tool is an wrapper for snscrape that pulls Tweets from a searched user and the tweets from a customizable number of people within their network. The tool also retrieves the tweet content, including some stats about the Tweet at the time of collection (e.g. nr of likes, retweets). The users that were initially searched for were @elonmusk and @ZelenskyyUa. The reasons behind this was that I wanted to try to answer the following:"),
              p("   1. Have most recent controversial tweets by the former had any impact that could be seen at the level of tweet metadata?"),
              p("   2. Has been any change in the tweet metadata of for the latter, especially after his response to @elonmusk's controversial twitter poll?"),
              p("   3. Can we use NLProc techniques to analyse the corpus of tweets and find any useful insights."),
              p("For the last point, @elonmusk is a good choice, since he has a tendency to tweet about a variety of divergent topics."),
              br(),
            
              #currently will not work, but will figure this out at some pt
              # fluidRow(
              #   box(pickerInput(
              #     'selected_disp_name', label='Which of the main users to show?', 
              #     choices= c(
              #       'ElonMusk',
              #       'ZelenskyyUa'
              #     ), 
              #     selected = c("ElonMusk"), 
              #     options = list(`actions-box` = TRUE),
              #     multiple = T
              #   ))
              # ), 
              
              fluidRow(
                box(pickerInput('tweet_stats_variable', label='Which Twitter Stats to show', 
                                choices = c('Number of likes', 
                                            'Number of retweets', 
                                            'Number of responses'
                                ),
                                selected=c("Number of responses"),
                                options = list(`actions-box` = TRUE),
                                multiple = T
                )), 
                box(pickerInput('by_or_at_musk_select', label='By @elonmusk/Mentions @elonmusk', 
                    choices = c('Mentions @elonmusk', 
                                'By @elonmusk'
                                ),
                    selected=c('By @elonmusk'),
                    options = list(`actions-box` = TRUE),
                    multiple = T
                    )
                ),
                box(textInput('search_term_tweets', label='Only show tweets with this in them:',
                              value="*")
                )
              ),
              
              fluidRow(
                box(plotlyOutput('TS_tweet_stats', height=450, width="100%"), width=12)
              ),
              
              
              
              # next row 
              fluidRow(
                box(plotlyOutput('before_and_after_mean_plot', height=300)), 
                box(plotOutput('before_and_after_boxplot', height=300))
              ),
              
  
              
              fluidRow(
                box(
                  # title='Accounts with most followers within dataset',
                  sliderInput('N_slider_followers', 'Who has the most followers: Number of accounts to show', 0, 100, 20),
                  height = 100
                ) ,
                box(
                  # title='Accounts with most followers within dataset',
                  sliderInput('N_slider_posts', 'Who has the most posts: Number of accounts to show', 0, 100, 20),
                  height = 100
                )
              ),
              
              fluidRow(
                box(plotlyOutput('followers_barplot')), 
                box(plotlyOutput('posts_barplot'))
              )
              
      ),
      
      
      # Third tab content
      tabItem(tabName = "topics",
              h2("Topic Modelling"),
              p('I used the open-source ', tags$a(href="https://github.com/digital-scrappy/network-analysis-hackathon", "BertTopic"), 'tool to mine the text data for emergent topics'),
              # more paragraphs here detailing experimentation, params, etc..
              p('The topics you see in the table and first plot were created automatically by BERTopic, except for a small amount of augmentation done manually by me after running the model. Originally there'),
              p('were more topics in the plot you can see (the table on the right shows all the original ones). There were a few dense clusters of topics with very similar meanings that were almost entirely'),
              p('overlapping on the plot - most notably topics 1, 10, 29 and 46, which are all related to Russia\'s invasion of Ukraine and its consequences). '),
              p('I used a pre-trained transformer model to encode the tweets as embeddings, then these were passed to BertTopic to create clusters and identify topics.'),
              br(),
              
              fluidRow(
                box(
                  htmlOutput('topics_dash', width=5),
                  title='Topic Analysis',
                  width=7
                ),
                box(
                  dataTableOutput('topics_table', width='100%'), width=5
                )
              ),
          
                p("@article{grootendorst2022bertopic"),
                p("        title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},"),
                p("        author={Grootendorst, Maarten}"),
                p("        journal={arXiv preprint arXiv:2203.05794},"),
                p("        year={2022}}")

      ),

      
      # Third tab content
      tabItem(tabName = "data",
              h2("The Cleaned Twitter Data"),
              fluidPage(
               mainPanel(  title='Clean_data', dataTableOutput( 'Clean_data', width='100%'), width=12)
               
              )
      )
    )
    
  
)
)





server <- function(input, output) {set.seed(122)
  
  
  ### DATA ETL AND WRANGLING HERE #########################################
  
  topics_fpath <- 'viz_topics_22_10_14_redux.html'
  get_html_viz<-function(fpath) {
    return(includeHTML(fpath))
  }
  
  output$topics_dash <- renderUI({get_html_viz(topics_fpath)})
  
  
  
  topics_df<-read.csv('../data/preds/topic_model_table.csv')
  topics_df <- topics_df[c("Topic","Count","Name")]
  topics_df <- subset(topics_df, Topic!=-1)
  topics_df <- topics_df[order(-topics_df$Count),]
  
  output$topics_table<-renderDataTable({
                                      topics_df
                                        }, 
                                        filter='top',
                                        rownames=FALSE,
                                        options = list(scrollX = TRUE)
                                        )
                                      
  
  
  # disp_df <- read.csv('../data/clean/clean_text_and_metadata.csv')
  disp_df <- read.csv('../data/clean/dashboard_data.csv')
  
  #getting user summary stats
  df <- read.csv('../data/raw/user_attributes.csv')
  df <- distinct(df) 
  head(df, 2)
  #subset for only those users in the main dashboard data
  df <- subset(df, X %in% disp_df$display_name)
  
  top_n <- 20
  # print(names(df))
  df<- df %>%
    dplyr::rename(
      Handle = X,
      Followers =  X.followers,
      Posts = X.posts,
      Friends = X.friends,
      Verified = verified,
      Display_name = display.name
    )
  # print(names(disp_df))
  disp_df<- disp_df %>%
    dplyr::rename(
      Display_name = display_name,
      Number_likes= X.likes,
      Number_retweets = X.retweets,
      Number_responses = X.responses
    )
  
  # will figure this out at some point
  # selected_account= observe({input$selected_disp_name} )
    
  df$Verified <- gsub('True', 'Verified', df$Verified)
  df$Verified <- gsub('False', 'Not verified', df$Verified)
  
  countfollow <- df[c('Handle', 'Followers','Posts', 'Verified')][order(df$Followers, decreasing=TRUE),]
  countpost <- df[c('Handle', 'Posts','Followers', 'Verified')][order(df$Posts, decreasing=TRUE),]

  
  print('Got counts of followers and posts')
  #now another dataframe which handles number of likes before and after event
  # target_df <- subset(disp_df,  grepl(targets, extracted_twitter_handles))
  # here we take the input, from the selector
  
  selected_account <- 'elonmusk'
  target_df <- subset(disp_df, Display_name==selected_account)
  
  # now groupby and sum
  print(names(target_df))
  target_mean <- target_df %>% group_by(Before_or_after_controversy) %>%
    summarise('Number of likes' = mean(Number_likes),
              'Number of retweets' = mean(Number_retweets),
              'Number of responses' = mean(Number_responses)
    )
  target_mean <- as.data.frame(target_mean)
  
  print('Created target_mean data')
  print(names(target_mean))
  
  target_mean <- target_mean %>% 
    dplyr::rename('Before or after controversial tweet' = Before_or_after_controversy)
  #melt it so it's easier to plot
  target_mean_melt <- reshape::melt(target_mean, id = c('Before or after controversial tweet'))
  target_mean_melt <- as.data.frame(target_mean_melt)
  target_mean_melt$`Before or after controversial tweet` <- factor(target_mean_melt$`Before or after controversial tweet`,
                                                                   levels = c('Before', 'After'))
  
  print('Created target_mean_melt')                                               
  # print(names(target_df))
  
  ## dataframe for boxplot
  target_df <- target_df %>%
    dplyr::rename( 'Number of likes' = 'Number_likes', 
            'Number of retweets' = 'Number_retweets', 
            'Number of responses' = 'Number_responses')
  
  
  target_df_melt<- reshape::melt(target_df[c('Number of likes', 
                                             'Number of retweets', 
                                             'Number of responses', 
                                             'Before_or_after_controversy') ], 
                                 id='Before_or_after_controversy')
  target_df_melt$Before_or_after_controversy<- factor(target_df_melt$Before_or_after_controversy, 
                                                      levels = c('Before', 'After'))
  print('Created target_df_melt')    
  target_df_melt <- target_df_melt %>% 
    dplyr::rename('Before or after controversial tweet' = Before_or_after_controversy)
  
  
  
  # time_target_df <- target_df
  time_target_melt<- read.csv('../data/viz/tweet_stats_over_time.csv')
  time_target_melt$datetime <- as.POSIXct(time_target_melt$datetime)
  
  # time_target_df <- time_target_df %>%
  #   select('DT', 'display_name', 
  #          'clean_tweet_text', 
  #          'Number of likes', 
  #          'Number of retweets', 
  #          'Number of responses')
  
  # time_target_df <- time_target_df %>%
  #   rename( 'Number of likes' = 'Number_likes', 
  #           'Number of retweets' = 'Number_retweets', 
  #           'Number of responses' = 'Number_responses')

    
  # time_target_melt<- reshape::melt(time_target_df, id.vars=c('DT', 'Display_name', 'clean_tweet_text'))
  
  # cutoff_date <- as.POSIXct('2022-10-03 16:15:43+0000')
  # wrapping teh text
  # time_target_melt$clean_tweet_text = lapply(time_target_melt$clean_tweet_text, function(x){stringr::str_wrap(x, 15)})
  # time_target_melt$clean_tweet_text = lapply(time_target_melt$clean_tweet_text, function(x){gsub('\n', '<br>', x)})
  
  # time_target_melt$clean_tweet_text
  
  
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
      layout(yaxis = list(categoryorder = "total ascending", 
                          title=NA), 
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
               subtitle = paste('How have the average stats for ', selected_account, '\'s Tweets changed\nafter his poll on the war?'),
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
      
      labs(title='Before and after the poll\n(excluding extreme outliers)', 
           subtitle = 'How have the average stats for Elon\'s Tweets\nchanged after his poll on the war?', 
           x = '', 
           y='')+
      scale_y_continuous(limits = quantile(target_df_melt$value, c(0.0, 0.9)))+
      theme_minimal()+
      coord_flip()+
      theme(legend.title=element_blank(),
            text = element_text(family = 'Open Sans', size=16, face='bold')
            # axis.text.x=element_text(angle=45, vjust=1.5, hjust=1)
            )
    
  })
  
  
  output$TS_tweet_stats <- renderPlotly({
    plot_ly(
      data = time_target_melt %>% dplyr::filter(grepl(input$search_term_tweets, tweet_text, ignore.case=TRUE)) %>%
        subset(variable==input$tweet_stats_variable) %>%
        subset(By_or_at_Musk==input$by_or_at_musk_select),
      x = ~datetime, 
      y = ~value,
      color=~variable,
      text=~tweet_text,
      hovertemplate = paste('<br><b>Date</b>:%{x}',
                            # '<br><b>%{color}</b>: '
                            '<br>%{y}',
                            '<br><b>Text</b>:%{text}'),
      type = 'scatter', mode='markers')  %>% 
      layout(yaxis=list(title=NA), 
             xaxis = list(title=NA), 
             title = list(title=paste("How have the Twitter changed in this period for either Musk or people mentioning him?"), 
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

    
