---
date: "2022-10-18"
external_link: ""
image: 
  caption: Snapshot of topics extracted from Tweets
  focal_point: Smart
links:
- icon: github
  icon_pack: fab
  name: Check out
  url: https://github.com/Ioana-P/twitter_sentiment_tracking
# slides: example
summary: Using OSINT tools and Transformers to extract topics and sentiment from Elon Musk's corner of the Twitter-sphere. 
tags: 
  - "Deep Learning"
  - "NLProc"
  
# output:
#   html_document:
#     includes:
#       before_body: fig/scatter_topics_SELECT.html

title: Visualizing and quantifying topics on Twitter
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""
---
# Visualizing and quantifying topics on Twitter


### Using OSINT tools and Transformers to extract topics and sentiment
Using the Blattodea tool that I helped develop during a hackathon, I retrieved the most recent tweets from Elon Musk. I then used one of [HuggingFace's](https://huggingface.co/models) pre-trained sentiment classification models and [BERTopic](https://maartengr.github.io/BERTopic/index.html) to extract and visualize key themes.
I have also developed an RShiny dashboard for this project to hone my interactive visualization skills. 

![Dashboard screenshot](fig/dash_screenshot.png)

### Results and thoughts
(Results are analysed and visualised at greater length inside the repo's index.Rmarkdown notebook. For cleaning, EDA and modelling code, please see the Jupyter notebooks in the repo, explained in the filing system below)

I have been able to extract clear and definite topics from the collected data, and the pattern of activity around key themes has been what I expected it to be. For example, Musk's opining on ending the war in Ukraine generated a larger amount of responses across the board than his other tweets. Through this project I've found that BERTopic has been extremely useful in extracting information from unstructured text data, and I plan on using it in future projects.

![Hierarchy of our topics of interest](fig/hierarchical_select_top.png)

Topics 29 and 9 form an understandable cluster together as there were a significant number of tweets focused on Starlink's activity in Ukraine (and its commercial activity more generally), both by Musk and his followers. 
The grouping of clusters 3 and 2 is more interesting: closer inspection of 2 revealed that it includes some tweets related to Twitter bots (something Musk has made a point of discussing openly recently), which would link it sensibly to topic 3; however some of the tweets were also referring to Tesla 'bots' (i.e. Tesla's robotics research and department). If we'd had a set of topic labels for each of these, it's very likely that BERTopic would misclassify the tweets in this particular topic. Topic two tweets range from :

| "Tesla Bot is the future of robotics 🤯 "

to 

| "If Twitter put as much effort into botspam removal as they do into subpoenas we wouldnt have this problem in the first place"

Human language users like us can tell that these are two distinct themes. Yet I'd say that this amalgamation of is quite an understandable mistake, given the amount of words that appear across topic 2 that make it strongly related to topic 3 and others. It is possible that if we had more documents, BERTopic would've split topic 2 into another leaf. 

Towards the bottom we can also see three topics that, although not as closely aligned as the war-themed ones, still cluster together. These are all connected by the theme of rocketry, SpaceX and all of Musk's space-related endeavours. It's encouraging to see that the model was able to place these closely together. Note the next leaf that joins this particular sub-branch, topic 55, related to engines. Now, when I originally looked at this topic, I figured that the algorithm had misgrouped again and that discussions of engines must be focussed on cars. However, I was wrong and the algo was right - inspection of the tweets with this as their main topic (currently topic nr 48 in our merged data) revealed that my intuition was wrong:

| "Hi Elon  😀 According to  380 tankers delivered to Starbase So I would be really happy to know Will B7 performs a long duration full stack 33 engines static fire Thanks🙏😀"

and

| "How certain are you on the final number of engines for Superheavy Booster"

and, a slight outlier in some sense:

| "there will be a day when there are literally 69 boosters 🤓"

I do enjoy points of childish levity in a dataset. 
*Anyway*, this is another great example of BERTopic's strengths as a model. 

The next diagrams shows the topics and their constituent topics, collapsed onto a 2-D plane with UMAP. It's very encouraging to see that the tweets we know to be distinct clusters (1,36,35,10 and 19,53) are still visibly separate even in this representation.

(If you wish to inspect this further there are two HTML files of this saved in the fig folder)

(I have deliverately not highlighted all of the topics of interest here because the plot would be far too cluttered them.)

### Repo filing system:

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


References:
@article{grootendorst2022bertopic,
  title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
  author={Grootendorst, Maarten},
  journal={arXiv preprint arXiv:2203.05794},
  year={2022}
}
