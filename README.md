# Visualizing and quantifying topics on Twitter


### Using OSINT tools and Transformers to extract topics and sentiment
##### Using the Blattodea tool that I helped develop during a hackathon, I retrieved the most recent tweets from Elon Musk. I then used HuggingFace\'s pre-trained sentiment analysis tool and BERTopic to extract and visualize key themes
##### I have also developed an RShiny dashboard for this project to hone my interactive visualization skills. 

![Snapshot of the documents collected and their algorithmically determined topics]('fig/snapshot_add_docs.png')

![Dashboard screenshot](fig/dash_screenshot.png)

### Filing system:

Notebooks
1. index.Rmarkdown - principal notebook of findings and final results; most relevant notebook to most people
2a. Topic_modelling_with_BERT.ipynb - notebook details the journey of analysing the model results and extracting insights.
2b. Modelling_w_BERTopic_GColab.ipynb - Google Colab notebook where the BERTopic and sentiment models' results were generated. 
3. EDA.ipynb - rough exploration of the data; go here for more in-depth look at some of the data. Most of the visualizations therein were not used.
4. data_cleaning.ipynb - notebook detailing the entire cleaning process. Primarily relies on py modules within functions.

Folders
* data - folder containing raw, processed, clean and feature data and any additional summary data generated. Also contains inferred data (i.e. tweets and their predicted sentiment; tweets and their associated topics)
* models - the fitted BERTopic model files - **NOTE** unfortunately, due to a bug in the way the models were saved, it is not possible to load them up locally on a lot of machines. However, it is possible to reproduce their creation and fitting on Google Colab, using the [Colab notebook](https://github.com/Ioana-P/IoanaFio/blob/main/content/project/twitter_sentiment_tracking/Modelling_w_BERTopic_GColab.ipynb) inside this repo
*fig - all data viz (including interactive HTML files)
*functions
* archive - any additional files and subfolders will be here
