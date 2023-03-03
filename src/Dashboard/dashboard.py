
from numpy import True_
import pandas as pd
pd.set_option('max_rows', 20)
import plotly.express as px 
import plotly.io as pio 
pio.renderers.default = 'browser'
import re
import sched, time
from multiprocessing import Process


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import snscrape.modules.twitter as sntwitter

import dash
from dash.dependencies import Input, Output
from dash import dcc, dash_table
from dash import html
from dash import callback_context
import dash_bootstrap_components as dbc
import plotly.express as px

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from io import BytesIO
import base64

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as DT

todayDate = DT.date.today()

todayNow = todayDate.strftime("%Y-%m-%d")
daysAgo = 3
weekAgo = todayDate - DT.timedelta(days=daysAgo)
weekAgo = weekAgo.strftime("%Y-%m-%d")

script_fn = 'sentiment.py'
exec(open(script_fn).read())

df = pd.read_csv('outputBank.csv')
dfMer = pd.read_csv('outputLynch.csv')

other = df.loc[df['Service'] == "Other"]
ATM = df.loc[df['Service'] == "ATM"]
app = df.loc[df['Service'] == "App"]
online = df.loc[df['Service'] == "online"]
 
ATMpos = ATM.loc[df['Sentiment'] == 'positive']
ATMneg = ATM.loc[df['Sentiment'] == 'negative']

appPos = app.loc[df['Sentiment'] == 'positive']
appNeg = app.loc[df['Sentiment'] == 'negative']

onlinePos = online.loc[df['Sentiment'] == 'positive']
onlineNeg = online.loc[df['Sentiment'] == 'negative']

otherPos = other.loc[df['Sentiment'] == 'positive']
otherPos = other.loc[df['Sentiment'] == 'negative']

if 'positive' in df['Sentiment'].unique():
    posTotal = df['Sentiment'].value_counts().positive

negTotal = df['Sentiment'].value_counts().negative
neutralTotal = df['Sentiment'].value_counts().neutral

otherMer = dfMer.loc[dfMer['Service'] == "Other"]
ATMMer = dfMer.loc[dfMer['Service'] == "ATM"]
appMer = dfMer.loc[dfMer['Service'] == "App"]
onlineMer = dfMer.loc[dfMer['Service'] == "online"]
 
ATMposMer = ATMMer.loc[dfMer['Sentiment'] == 'positive']
ATMnegMer = ATMMer.loc[dfMer['Sentiment'] == 'negative']

appPosMer = appMer.loc[dfMer['Sentiment'] == 'positive']
appNegMer = appMer.loc[dfMer['Sentiment'] == 'negative']

onlinePosMer = onlineMer.loc[dfMer['Sentiment'] == 'positive']
onlineNegMer  = onlineMer.loc[dfMer['Sentiment'] == 'negative']

otherPosMer = otherMer.loc[dfMer['Sentiment'] == 'positive']
otherPosMer = otherMer.loc[dfMer['Sentiment'] == 'negative']

if 'positive' in dfMer['Sentiment'].unique():
    posTotalMer = dfMer['Sentiment'].value_counts().positive

negTotalMer = dfMer['Sentiment'].value_counts().negative

if 'neutral' in dfMer['Sentiment'].unique():
    neutralTotalMer = dfMer['Sentiment'].value_counts().neutral



def getTweets(company): 
    tweet_list = []

    if company == 'Bank of America':
        atUser = "@BankOfAmerica"
    if company == 'Merrill Lynch':
        atUser = '@MerrillLynch'
    else:
        atUser = company
    
    #+ ' until:' + todayNow

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i, tweet in enumerate(
            sntwitter.TwitterSearchScraper(atUser +  ' since:' + weekAgo).get_items()):
        tweet_list.append([tweet.date, tweet.id, tweet.content.replace("\n", ""), tweet.user.location.replace("\n", ""),
                        tweet.user.username])

    # Creates dataframe of scraped tweeets
    all_tweets = pd.DataFrame(tweet_list, columns=['Timestamp', 'Tweet ID', 'Tweet', 'Location', 'Username'])


    # Creates CSV file with BoA Tweets scraped from Twitter
    if company == 'Bank of America':
        all_tweets.to_csv('BoA-tweets.csv', sep=',', index=False)
    if company == 'Merrill Lynch':
        all_tweets.to_csv('ML-tweets.csv', sep=',', index=False)

    
    

def sentimentCreation():
    df = pd.read_csv('newSent.csv')

    bankAmericaTweets = pd.read_csv('BoA-tweets.csv')
    merillTweets = pd.read_csv('ML-tweets.csv')

    timestampBOA = bankAmericaTweets.iloc[:,0]
    boaTweetID = bankAmericaTweets.iloc[:,1]
    boaTweets = bankAmericaTweets.iloc[:,2]
    locations = bankAmericaTweets.iloc[:,3]
    usernames = bankAmericaTweets.iloc[:,4]


    timestampMer = merillTweets.iloc[:,0]
    merTweetID = merillTweets.iloc[:,1]
    merTweets = merillTweets.iloc[:,2]
    locationsMer = merillTweets.iloc[:,3]
    usernamesMer = merillTweets.iloc[:,4]


    sentiment = df['Sentiment']
    tweet = df['Tweet']
    
    tfidf = TfidfVectorizer(max_features=53, ngram_range=(3,3))
    tweeted = tfidf.fit_transform(tweet)



    sentiment_train, sentiment_test, tweet_train, tweet_test = train_test_split(sentiment, tweeted, test_size=0.2, random_state=0)
    clf = LinearSVC()
    clf.fit(tweet_train, sentiment_train)



    header = ['Sentiment', 'Tweet', 'Service']
    tweets = []
    sentiment = []
    service = []

    headerMer = ['sentiment', 'tweet', 'service']
    tweetsLynch = []
    sentimentLynch = []
    serviceLynch = []

    positive = '3'
    negative = '1'
    neutral = '2'


    aTM = 'ATM'
    mobile = "App"
    online = "online"
    other = "Other"


    atmList = ['ATM', 'dispense','cash machine', 'atm machine']
    mobileList = ['mobile','phone','app']
    onlineList = ['online','web','site','website','web site', 'online banking']

    for x in boaTweets:


        vec = tfidf.transform([x])
        clf.predict(vec)
        tweets.append(x)

        # print(clf.predict(vec))
        if clf.predict(vec) == [3]:
            sentiment.append(positive)

        if clf.predict(vec) == [1]:
            sentiment.append(negative)

        if clf.predict(vec) == [2]:
            sentiment.append(neutral)
    

        if [ele for ele in atmList if(ele in x)]:
            service.append(aTM)

        elif [ele for ele in mobileList if(ele in x)]:
            service.append(mobile)

        elif [ele for ele in onlineList if(ele in x)]:
            service.append(online)
        else:
            service.append(other)
        
    
    for x in merTweets:


        vec = tfidf.transform([x])
        clf.predict(vec)
        tweetsLynch.append(x)

        # print(clf.predict(vec))
        if clf.predict(vec) == [3]:
            sentimentLynch.append(positive)

        if clf.predict(vec) == [1]:
            sentimentLynch.append(negative)

        if clf.predict(vec) == [2]:
            sentimentLynch.append(neutral)
    

        if [ele for ele in atmList if(ele in x)]:
            serviceLynch.append(aTM)

        elif [ele for ele in mobileList if(ele in x)]:
            serviceLynch.append(mobile)

        elif [ele for ele in onlineList if(ele in x)]:
            serviceLynch.append(online)
        else:
            serviceLynch.append(other)


    data = {'Sentiment':sentiment,'Service': service,'Tweet':tweets, 'Timestamp': timestampBOA, 'Location':locations}

    dataMer = {'Sentiment':sentimentLynch,'Service': serviceLynch,'Tweet':tweetsLynch, 'Timestamp': timestampMer, 'Location':locationsMer}

    outputBOA = pd.DataFrame(data)

    outputBOA['Sentiment'] = outputBOA['Sentiment'].replace('1','negative')
    outputBOA['Sentiment'] = outputBOA['Sentiment'].replace('3','positive')
    outputBOA['Sentiment'] = outputBOA['Sentiment'].replace('2','neutral')

    

    outputBOA.to_csv('outputBank.csv',mode='w',header=True,index=False)



    outputMer = pd.DataFrame(dataMer)

    outputMer['Sentiment'] = outputMer['Sentiment'].replace('1','negative')
    outputMer['Sentiment'] = outputMer['Sentiment'].replace('3','positive')
    outputMer['Sentiment'] = outputMer['Sentiment'].replace('2','neutral')


    outputMer.to_csv('outputLynch.csv',mode='w',header=True,index=False)


external_stylesheets = [dbc.themes.BOOTSTRAP]



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Bank of America Sentiment'


colorsBOA = {
    'background':'#b50000',
    'bodyColor':'#fffff',
    'text': '##fffff',
    'body':'#fffff'
}

colorsMer = {
    'background':'#00378f',
    'bodyColor':'#fffff',
    'text': '##fffff',
    'body':'#fffff'
}

def get_page_heading_style(company):
    if company == 'BOA':
        return {'backgroundColor': colorsBOA['background']}
    elif company == 'ML':
        return {'backgroundColor': colorsMer['background']}



def get_page_heading_title(company):
    if company == 'BOA':
        titleName = 'Bank of America'
    if company == 'ML':
        titleName = 'Merrill Lynch'
    return html.H1(children=titleName,
        style={
        'textAlign': 'center',
        'color':'#fff'
    })

def get_page_heading_subtitle():
    return html.Div(children='',
        style={
            'textAlign':'center',
            'color':'#fff'
        })

def generate_page_header(company):
    main_header =  dbc.Row(
        [
            dbc.Col(get_page_heading_title(company),md=12)
        ],
        align="center",
        style=get_page_heading_style(company)
    )
    subtitle_header = dbc.Row(
        [
            dbc.Col(get_page_heading_subtitle(),md=12)
        ],
        align="center",
        style=get_page_heading_style(company)
    )
    header = (main_header,subtitle_header)
    return header


def generate_card_content(card_header,card_value,overall_value):
    card_head_style = {'textAlign':'center','fontSize':'150%'}
    card_body_style = {'textAlign':'center','fontSize':'200%'}
    card_header = dbc.CardHeader(card_header,style=card_head_style)
    card_body = dbc.CardBody(
        [
            html.H5(f"{int(card_value):,}", className="card-title",style=card_body_style),
            html.P(
                "Total: {:,}".format(overall_value),
                className="card-text",style={'textAlign':'center'}
            ),
        ]
    )
    card = [card_header,card_body]
    return card

def generate_cards():

    df = pd.read_csv('outputBank.csv')

    posTotal = df['Sentiment'].value_counts().positive
    negTotal = df['Sentiment'].value_counts().negative
    neutralTotal = df['Sentiment'].value_counts().neutral

    totalRows = len(df.index)
    cards = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card(generate_card_content("Positive",posTotal,totalRows), color="success", inverse=True),md=dict(size=2,offset=3)),
                    dbc.Col(dbc.Card(generate_card_content("Negative",negTotal,totalRows), color="danger", inverse=True),md=dict(size=2)),
                    dbc.Col(dbc.Card(generate_card_content("Neutral",neutralTotal,totalRows),color="primary", inverse=True),md=dict(size=2)),
                ],
                className="mb-4",
            ),
        ],id='card1'
    )
    return cards


def generate_card_contentMer(card_header,card_value,overall_value):
    card_head_style = {'textAlign':'center','fontSize':'150%'}
    card_body_style = {'textAlign':'center','fontSize':'200%'}
    card_header = dbc.CardHeader(card_header,style=card_head_style)
    card_body = dbc.CardBody(
        [
            html.H5(f"{int(card_value):,}", className="card-title",style=card_body_style),
            html.P(
                "Total: {:,}".format(overall_value),
                className="card-text",style={'textAlign':'center'}
            ),
        ]
    )
    card = [card_header,card_body]
    return card

def generate_cardsMer():

    dfMer = pd.read_csv('outputLynch.csv')

    posTotalMer = 0
    negTotalMer = 0
    neutralTotalMer = 0

    if 'positive' in dfMer['Sentiment'].unique():
        posTotalMer = dfMer['Sentiment'].value_counts().positive

    if 'negative' in dfMer['Sentiment'].unique():
        negTotalMer = dfMer['Sentiment'].value_counts().negative
    
    if 'neutral' in dfMer['Sentiment'].unique():
        neutralTotalMer = dfMer['Sentiment'].value_counts().neutral

    totalRows = len(dfMer.index)

    cards = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card(generate_card_contentMer("Positive",posTotalMer,totalRows), color="success", inverse=True),md=dict(size=2,offset=3)),
                    dbc.Col(dbc.Card(generate_card_contentMer("Negative",negTotalMer,totalRows), color="danger", inverse=True),md=dict(size=2)),
                    dbc.Col(dbc.Card(generate_card_contentMer("Neutral",neutralTotalMer,totalRows),color="primary", inverse=True),md=dict(size=2)),
                ],
                className="mb-4",
            ),
        ],id='card1'
    )
    return cards


tweetPos = df.loc[df['Sentiment'] == "positive"]
tweetNeg = df.loc[df['Sentiment'] == "negative"]
tweetNeut = df.loc[df['Sentiment'] == "neutral"]

tweetWordsPos = tweetPos['Tweet']
tweetWordsNeg = tweetNeg['Tweet']
tweetWordsNeut = tweetNeut['Tweet']


tweetPosMer = dfMer.loc[dfMer['Sentiment'] == "positive"]
tweetNegMer = dfMer.loc[dfMer['Sentiment'] == "negative"]
tweetNeutMer = dfMer.loc[dfMer['Sentiment'] == "neutral"]

tweetWordsPosMer = tweetPosMer['Tweet']
tweetWordsNegMer = tweetNegMer['Tweet']
tweetWordsNeutMer = tweetNeutMer['Tweet']


stopwords = ['what', 'who', 'is', 'a', 'at', 'is', 'he', '@MerrillLynch', '@BankofAmerica', 'the', 'I','and', 'to', 'i', 'of', 'you', 'for', 'in', 'this', 'that', 'me', 'will', 'have','his','your','on','my', 'with', 'are','be','was','it','they','been','get','has','http','https','x80','xe2','tco','xa6','xf0','t','co']


sentLabels = []
sentLabelsMer = []
colorSequence = []
colorSequenceMer = []


if 'positive' in df['Sentiment'].unique():
    sentLabels.append('positive')
    colorSequence.append('green')

if 'neutral' in df['Sentiment'].unique():
    sentLabels.append('neutral')
    colorSequence.append('blue')

if 'negative' in df['Sentiment'].unique():
    sentLabels.append('negative')
    colorSequence.append('red')

if 'positive' in dfMer['Sentiment'].unique():
    sentLabelsMer.append('positive')
    colorSequenceMer.append('green')

if 'neutral' in dfMer['Sentiment'].unique():
    sentLabelsMer.append('neutral')
    colorSequenceMer.append('blue')


if 'negative' in dfMer['Sentiment'].unique():
    sentLabelsMer.append('negative')
    colorSequenceMer.append('red')

pie = px.pie(df['Sentiment'].value_counts().sort_values(ascending=True),values='Sentiment', names=sentLabels, color_discrete_sequence=['red','blue','green'])

pieMer = px.pie(dfMer['Sentiment'].value_counts().sort_values(ascending=True),values='Sentiment', names=sentLabelsMer, color_discrete_sequence=['red','blue','green'])





forTable = df.copy()

forTable['Tweet'] =  [re.sub('@[^\s]+','', str(x)) for x in forTable['Tweet']]

forTable['Tweet'] =  [re.sub(r"http\S+", '', str(x)) for x in forTable['Tweet']]

tableGenerated = forTable.drop(columns=['Service', 'Sentiment', 'Timestamp'])

def generate_table(dataframe, max_rows=20):


    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

fig = make_subplots(
    rows=1, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    specs=[[{"type": "table"}]]
)

fig.add_trace(
    go.Table(
        header=dict(
            values=["Service", "Tweet", "Timestamp", "Location"],
            font=dict(size=10),
            align="left"
        ),
        cells=dict(
            values=[df[k].tolist() for k in df.columns[1:]],
            align = "left")
    ),
    row=1, col=1
)
fig.update_layout(
    height=1200,
    width=1800,
    showlegend=False,
    title_text="Recent tweets: ",
)

figMer = make_subplots(
    rows=1, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    specs=[[{"type": "table"}]]
)

figMer.add_trace(
    go.Table(
        header=dict(
            values=["Service", "Tweet", "Timestamp", "Location"],
            font=dict(size=10),
            align="left"
        ),
        cells=dict(
            values=[dfMer[k].tolist() for k in dfMer.columns[1:]],
            align = "left")
    ),
    row=1, col=1
)
figMer.update_layout(
    height=1200,
    width=1800,
    showlegend=False,
    title_text="Recent tweets: ",
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


headerTitle = 'Data Shown for Past ' + str(daysAgo) + ' Days'

def generate_layout(): 
    page_header = generate_page_header('BOA')
    layout = dbc.Container(
        [
            page_header[0],
            page_header[1],
            html.Hr(style={'color':'#fff'}),
            html.Div(id='top-cards'),
            html.H3(children=headerTitle,style={'textAlign': 'center'}),
            dcc.Interval(
            id='interval-cards',
            interval=1*60000, # in milliseconds
            n_intervals=0),
            html.A(html.Button('Merrill Lynch', style={'background-color': 'rgb(0, 55, 143)','color':'white','border':'2px solid #fff', 'height':'60px','width':'100px'}),href='/merrill-lynch'),
            html.Hr(style={'color':'#fff'}),
                html.Div(id='ratioPosNeg', children=[
                html.H6(children='Positive / Negative Ratio'),
                dcc.Graph(id='update-graph1'),
            dcc.Interval(
            id='interval-1',
            interval=1*60000, # in milliseconds
            n_intervals=0),
                html.H6(children='Category Sentiments: '),
                dbc.Tabs([
                dbc.Tab(label='ATM', tab_id='tab-atm-graph'),
                dbc.Tab(label='Mobile', tab_id='tab-mobile-graph'),
                dbc.Tab(label='Online', tab_id='tab-online-graph'),
                dbc.Tab(label='Other', tab_id='tab-other-graph'),
                ],
                id="tabs-cat",
                active_tab='tab-atm-graph'),
                html.Div(id='tabs-categories')
            ], style={'width': '50%', 'display': 'inline-block'}),
             dcc.Interval(
            id='interval-cats',
            interval=1*60000, # in milliseconds
            n_intervals=0),
            html.Div([html.H6(children='Word Cloud: '),
                dbc.Tabs([
                dbc.Tab(label='Positive', tab_id='tab-1-example-graph'),
                dbc.Tab(label='Negative', tab_id='tab-2-example-graph'),
                dbc.Tab(label='Neutral', tab_id='tab-3-example-graph'),
                ],
                id="tabs-example-graph",
                active_tab='tab-1-example-graph'),
                html.Div(id='tabs-content-example-graph'),
        dcc.Interval(
            id='interval-wc',
            interval=1*60000, # in milliseconds
            n_intervals=0),
                ],
                style={'width': '49%', 'display': 'inline-block', 'vertical-align':'top'}),
                html.Div(id='outputRefresh'),
                dcc.Graph(id='table-graph'),
            dcc.Interval(
            id='interval-table',
            interval=1*60000, # in milliseconds
            n_intervals=0),   
                ],fluid=True,style={'backgroundColor': colorsBOA['bodyColor']}
    )
    return layout

def index_page():
    return html.Div([
    html.A(html.Button('Merrill Lynch', style={'background-color': 'rgb(0, 55, 143)','color':'white','border':'2px solid #fff', 'height':'60px','width':'100px'}),href='/merrill-lynch'),
    html.Br(),
            html.A(html.Button('Bank of America', style={'background-color': 'rgb(181, 0, 0)','color':'white','border':'2px solid #fff', 'height':'60px','width':'100px'}),href='/boa')])

def generate_second(): 
    page_header = generate_page_header('ML')
    layout = dbc.Container(
        [
            page_header[0],
            page_header[1],
            html.Hr(style={'color':'#fff'}),
            html.Div(id='top-cardsMer'),
            html.H3(children=headerTitle,style={'textAlign': 'center'}),
            dcc.Interval(
            id='interval-cardsMer',
            interval=1*60000, # in milliseconds
            n_intervals=0),
            html.A(html.Button('Bank Of America', style={'background-color': 'rgb(181, 0, 0)','color':'white','border':'2px solid #fff', 'height':'60px','width':'100px'}),href='/boa'),
            html.Hr(style={'color':'#fff'}),
                html.Div(id='ratioPosNeg', children=[
                html.H6(children='Positive / Negative Ratio'),
                dcc.Graph(id='update-graph1Mer'),
            dcc.Interval(
            id='interval-1Mer',
            interval=1*60000, # in milliseconds
            n_intervals=0),
                html.H6(children='Category Sentiments: '),
                dbc.Tabs([
                dbc.Tab(label='ATM', tab_id='tab-atm-graphMer'),
                dbc.Tab(label='Mobile', tab_id='tab-mobile-graphMer'),
                dbc.Tab(label='Online', tab_id='tab-online-graphMer'),
                dbc.Tab(label='Other', tab_id='tab-other-graphMer'),
                ], 
                id="tabs-catMer",
                active_tab='tab-atm-graphMer'),
                html.Div(id='tabs-categoriesMer')
            ], style={'width': '50%', 'display': 'inline-block'}),
            dcc.Interval(
            id='interval-catsMer',
            interval=1*60000, # in milliseconds
            n_intervals=0),
            html.Div([html.H6(children='Word Cloud: '),
                dbc.Tabs([
                dbc.Tab(label='Positive', tab_id='tab-1-example-graphMer'),
                dbc.Tab(label='Negative', tab_id='tab-2-example-graphMer'),
                dbc.Tab(label='Neutral', tab_id='tab-3-example-graphMer'),
                ],
                id="tabs-example-graphMer",
                active_tab="tab-1-example-graphMer"),
                html.Div(id='tabs-content-example-graphMer'),
            dcc.Interval(
            id='interval-wcMer',
            interval=1*60000, # in milliseconds
            n_intervals=0),   
                ],
                style={'width': '49%', 'display': 'inline-block', 'vertical-align':'top'}),
                html.Div(id='outputRefresh'),
                dcc.Graph(id='table-graphMer'),
            dcc.Interval(
            id='interval-tableMer',
            interval=1*60000, # in milliseconds
            n_intervals=0),   
                ],fluid=True,style={'backgroundColor': colorsMer['bodyColor']}
    )
    return layout





@app.callback(Output('tabs-content-example-graph', 'children'),
            Input('tabs-example-graph', 'active_tab'),
            Input('interval-wc', 'n_intervals'),)

                  
def wordCloud(tab, n):

    df = pd.read_csv('outputBank.csv')

    tweetPos = df.loc[df['Sentiment'] == "positive"]
    tweetNeg = df.loc[df['Sentiment'] == "negative"]
    tweetNeut = df.loc[df['Sentiment'] == "neutral"]

    tweetWordsPos = tweetPos['Tweet']
    tweetWordsNeg = tweetNeg['Tweet']
    tweetWordsNeut = tweetNeut['Tweet']

    stopwords = ['what', 'who', 'is', 'a', 'at', 'is', 'he', '@MerrillLynch', '@BankofAmerica', 'the', 'I','and', 'to', 'i', 'of', 'you', 'for', 'in', 'this', 'that', 'me', 'will', 'have','his','your','on','my', 'with', 'are','be','was','it','they','been','get','has','http','https','x80','xe2','tco','xa6','xf0','t','co']

    newPos = []

    for x in tweetWordsPos:
        newTweetsPos = x.replace("b'","")

        newPos.append(newTweetsPos)

    temp = [wrd for sub in newPos for wrd in sub.split()]

    resultPos  = [word for word in temp if word.lower() not in stopwords]



    allPos = ' '.join(resultPos)

    newNeg = []

    for x in tweetWordsNeg:
        newTweetsNeg = x.replace("b'","")

        newNeg.append(newTweetsNeg)

    temp2 = [wrd for sub in newNeg for wrd in sub.split()]

    resultNeg  = [word for word in temp2 if word.lower() not in stopwords]



    allNeg = ' '.join(resultNeg)

    newNeut = []

    for x in tweetWordsNeut:
        newTweetsNeut = x.replace("b'","")

        newNeut.append(newTweetsNeut)

    temp3 = [wrd for sub in newNeut for wrd in sub.split()]

    resultNeut  = [word for word in temp3 if word.lower() not in stopwords]



    allNeut = ' '.join(resultNeut)



    allPos =  re.sub('@[^\s]+','', allPos)
    allPos = re.sub(r"http\S+", '', allPos)

    allNeg =  re.sub('@[^\s]+','', allNeg)
    allNeg = re.sub(r"http\S+", '', allNeg)

    allNeut =  re.sub('@[^\s]+','', allNeut)
    allNeut = re.sub(r"http\S+", '', allNeut)


    wordcloudPos = WordCloud(max_words=100,width = 800, height = 900,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(allPos)


    wc_imgPos = wordcloudPos.to_image()

    wordcloudNeg = WordCloud(max_words=100,width = 800, height = 900,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(allNeg)

    wordcloudNeut = WordCloud(max_words=100,width = 800, height = 900,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(allNeut)

    wc_imgNeut = wordcloudNeut.to_image()

    wc_imgNeg = wordcloudNeg.to_image()

    with BytesIO() as buffer:
        wc_imgPos.save(buffer, 'png')
        img2 = base64.b64encode(buffer.getvalue()).decode()

    with BytesIO() as buffer:
        wc_imgNeg.save(buffer, 'png')
        img3 = base64.b64encode(buffer.getvalue()).decode()

    with BytesIO() as buffer:
        wc_imgNeut.save(buffer, 'png')
        img4 = base64.b64encode(buffer.getvalue()).decode()

    if tab == 'tab-1-example-graph':
        return  html.Div(children=[
                    html.Img(src="data:image/png;base64," + img2)
                ], style={'textAlign': 'center'})

    elif tab == 'tab-2-example-graph':
        return html.Div(children=[
                    html.Img(src="data:image/png;base64," + img3)
                ], style={'textAlign': 'center'})

    elif tab == 'tab-3-example-graph':
        return html.Div(children=[
                    html.Img(src="data:image/png;base64," + img4)
                ], style={'textAlign': 'center'})


@app.callback(Output('tabs-content-example-graphMer', 'children'),
            Input('tabs-example-graphMer', 'active_tab'),
            Input('interval-wcMer', 'n_intervals'),)

                    
def wordCloudMer(tab,n):

    dfMer = pd.read_csv('outputLynch.csv')


    tweetPosMer = dfMer.loc[dfMer['Sentiment'] == "positive"]
    tweetNegMer = dfMer.loc[dfMer['Sentiment'] == "negative"]
    tweetNeutMer = dfMer.loc[dfMer['Sentiment'] == "neutral"]

    tweetWordsPosMer = tweetPosMer['Tweet']
    tweetWordsNegMer = tweetNegMer['Tweet']
    tweetWordsNeutMer = tweetNeutMer['Tweet']

    newPosMer = []

    for x in tweetWordsPosMer:
        newTweetsPosMer = x.replace("b'","")

        newPosMer.append(newTweetsPosMer)

    tempMer = [wrdMer for subMer in newPosMer for wrdMer in subMer.split()]

    resultPosMer  = [wordMer for wordMer in tempMer if wordMer.lower() not in stopwords]


    allPosMer = ' '.join(resultPosMer)

    newNegMer = []

    for x in tweetWordsNegMer:
        newTweetsNegMer = x.replace("b'","")

        newNegMer.append(newTweetsNegMer)

    temp2Mer = [wrdMer for subMer in newNegMer for wrdMer in subMer.split()]

    resultNegMer  = [wordMer for wordMer in temp2Mer if wordMer.lower() not in stopwords]

    allNegMer = ' '.join(resultNegMer)

    newNeutMer = []

    for x in tweetWordsNeutMer:
        newTweetsNeutMer = x.replace("b'","")

        newNeutMer.append(newTweetsNeutMer)

    temp3Mer = [wrdMer for subMer in newNeutMer for wrdMer in subMer.split()]

    resultNeutMer  = [wordMer for wordMer in temp3Mer if wordMer.lower() not in stopwords]

    allNeutMer = ' '.join(resultNeutMer)


    allPosMer =  re.sub('@[^\s]+','', allPosMer)
    allPosMer = re.sub(r"http\S+", '', allPosMer)

    allNegMer =  re.sub('@[^\s]+','', allNegMer)
    allNegMer = re.sub(r"http\S+", '', allNegMer)

    allNeutMer =  re.sub('@[^\s]+','', allNeutMer)
    allNeutMer = re.sub(r"http\S+", '', allNeutMer)

    if len(allPosMer) == 0:
        allPosMer = 'No data available right now.'

    if len(allNegMer) == 0:
        allNegMer = 'No data available right now.'

    if len(allNeutMer) == 0:
            allNeutMer = 'No data available right now.'



    wordcloudPosMer = WordCloud(max_words=100,width = 800, height = 900,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(allPosMer)


    wc_imgPosMer = wordcloudPosMer.to_image()

    wordcloudNegMer = WordCloud(max_words=100,width = 800, height = 900,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(allNegMer)

    wordcloudNeutMer = WordCloud(max_words=100,width = 800, height = 900,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(allNeutMer)

    wc_imgNeutMer = wordcloudNeutMer.to_image()

    wc_imgNegMer = wordcloudNegMer.to_image()

    with BytesIO() as bufferMer:
        wc_imgPosMer.save(bufferMer, 'png')
        img5 = base64.b64encode(bufferMer.getvalue()).decode()

    with BytesIO() as bufferMer:
        wc_imgNegMer.save(bufferMer, 'png')
        img6 = base64.b64encode(bufferMer.getvalue()).decode()

    with BytesIO() as bufferMer:
        wc_imgNeutMer.save(bufferMer, 'png')
        img7 = base64.b64encode(bufferMer.getvalue()).decode()
        
    if tab == 'tab-1-example-graphMer':
        return  html.Div(children=[
                    html.Img(src="data:image/png;base64," + img5)
                ], style={'textAlign': 'center'})

    elif tab == 'tab-2-example-graphMer':
        return html.Div(children=[
                    html.Img(src="data:image/png;base64," + img6)
                ], style={'textAlign': 'center'})

    elif tab == 'tab-3-example-graphMer':
        return html.Div(children=[
                    html.Img(src="data:image/png;base64," + img7)
                ], style={'textAlign': 'center'})



@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])

def display_page(pathname):
    if pathname == '/boa':
        return generate_layout()
    elif pathname == '/merrill-lynch':
        return generate_second()
    else:
        return index_page()


@app.callback(Output('tabs-categories','children'),
            Input('tabs-cat','active_tab'),
            Input('interval-cats','n_intervals'))


def render_cats(catType,n):

    df = pd.read_csv('outputBank.csv')

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df['Timestamp'] = df['Timestamp'].dt.date
   

    lineMobe = df.loc[df['Service'] == 'App']
    lineMob = lineMobe.groupby(['Timestamp', 'Sentiment']).size().unstack('Sentiment')
    lineMob = lineMob.fillna(0)

    lineMobile = px.line(lineMob, x=lineMobe['Timestamp'].unique(), y=lineMob.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })

    lineOnle = df.loc[df['Service'] == 'online']
    lineOnl = lineOnle.groupby(['Timestamp', 'Sentiment']).size().unstack('Sentiment')
    lineOnl = lineOnl.fillna(0)

    lineOnline = px.line(lineOnl, x=lineOnle['Timestamp'].unique(), y=lineOnl.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     } )

    lineOthe = df.loc[df['Service'] == 'Other']
    lineOth = lineOthe.groupby(['Timestamp', 'Sentiment']).size().unstack('Sentiment') 
    lineOth = lineOth.fillna(0)

    lineOther = px.line(lineOth, x=lineOthe['Timestamp'].unique(), y=lineOth.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })

    
    lineATe = df.loc[df['Service'] == 'ATM']
    lineAT = lineATe.groupby(['Timestamp', 'Sentiment']).size().unstack('Sentiment')
    lineAT = lineAT.fillna(0)
    
    lineATM = px.line(lineAT, x=lineATe['Timestamp'].unique(), y=lineAT.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })


    if len(lineATe['Timestamp'].unique()) == 0:
        lineATM = px.line(lineAT, x=df['Timestamp'].unique(), y=lineAT.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })

    else:
        lineATM = px.line(lineAT, x=lineATe['Timestamp'].unique(), y=lineAT.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })
    
    if len(lineOnle['Timestamp'].unique()) == 0:
        lineOnline = px.line(lineOnl, x=df['Timestamp'].unique(), y=lineOnl.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })

    else:
        lineOnline = px.line(lineOnl, x=lineOnle['Timestamp'].unique(), y=lineOnl.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })
    
    if len(lineMobe['Timestamp'].unique()) == 0:
        lineMobile = px.line(lineMob, x=df['Timestamp'].unique(), y=lineMob.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })

    else:
        lineMobile = px.line(lineMob, x=lineMobe['Timestamp'].unique(), y=lineMob.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })


    if len(lineOthe['Timestamp'].unique()) == 0:
        lineOther = px.line(lineOth, x=df['Timestamp'].unique(), y=lineOth.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })

    else:
        lineOther = px.line(lineOth, x=lineOthe['Timestamp'].unique(), y=lineOth.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })


    if catType == 'tab-mobile-graph':
        return  dcc.Graph(figure=lineMobile)       


    elif catType == 'tab-online-graph':
        return  dcc.Graph(figure=lineOnline)       


    elif catType == 'tab-other-graph':
        return  dcc.Graph(figure=lineOther)       


    elif catType == 'tab-atm-graph':
        return  dcc.Graph(figure=lineATM)       

                
@app.callback(Output('tabs-categoriesMer','children'),
            Input('tabs-catMer','active_tab'),
            Input('interval-catsMer','n_intervals'))


def render_catsMer(catType,n):

    dfMer = pd.read_csv('outputLynch.csv')

    dfMer['Timestamp'] = pd.to_datetime(dfMer['Timestamp'])

    dfMer['Timestamp'] = dfMer['Timestamp'].dt.date
   

    lineMobe = dfMer.loc[dfMer['Service'] == 'App']
    lineMob = lineMobe.groupby(['Timestamp', 'Sentiment']).size().unstack('Sentiment')
    lineMob = lineMob.fillna(0)

    lineOnle = dfMer.loc[dfMer['Service'] == 'online']
    lineOnl = lineOnle.groupby(['Timestamp', 'Sentiment']).size().unstack('Sentiment')
    lineOnl = lineOnl.fillna(0)

    lineOthe = dfMer.loc[dfMer['Service'] == 'Other']
    lineOth = lineOthe.groupby(['Timestamp', 'Sentiment']).size().unstack('Sentiment') 
    lineOth = lineOth.fillna(0)

    lineATe = dfMer.loc[dfMer['Service'] == 'ATM']
    lineAT = lineATe.groupby(['Timestamp', 'Sentiment']).size().unstack('Sentiment')
    lineAT = lineAT.fillna(0)


    if len(lineATe['Timestamp'].unique()) == 0:
        lineATM = px.line(lineAT, x=dfMer['Timestamp'].unique(), y=lineAT.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })

    else:
        lineATM = px.line(lineAT, x=lineATe['Timestamp'].unique(), y=lineAT.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })
    
    if len(lineOnle['Timestamp'].unique()) == 0:
        lineOnline = px.line(lineOnl, x=dfMer['Timestamp'].unique(), y=lineOnl.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })

    else:
        lineOnline = px.line(lineOnl, x=lineOnle['Timestamp'].unique(), y=lineOnl.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })

    if len(lineMobe['Timestamp'].unique()) == 0:
        lineMobile = px.line(lineMob, x=dfMer['Timestamp'].unique(), y=lineMob.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })

    else:
        lineMobile = px.line(lineMob, x=lineMobe['Timestamp'].unique(), y=lineMob.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })


    if len(lineOthe['Timestamp'].unique()) == 0:
        lineOther = px.line(lineOth, x=dfMer['Timestamp'].unique(), y=lineOth.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })

    else:
        lineOther = px.line(lineOth, x=lineOthe['Timestamp'].unique(), y=lineOth.columns, color='Sentiment', color_discrete_map={'positive':'green', 
                          'neutral':'blue',
                          'negative':'red',
                     })


    if catType == 'tab-mobile-graphMer':
        return  dcc.Graph(figure=lineMobile)       


    elif catType == 'tab-online-graphMer':
        return  dcc.Graph(figure=lineOnline)       


    elif catType == 'tab-other-graphMer':
        return  dcc.Graph(figure=lineOther)       


    elif catType == 'tab-atm-graphMer':
        return  dcc.Graph(figure=lineATM)       




s = sched.scheduler(time.time, time.sleep)



@app.callback(Output('update-graph1', 'figure'),
            Input('interval-1', 'n_intervals'),)


def pieGraph (n):
    df = pd.read_csv('outputBank.csv')
    pieGraph = px.pie(df['Sentiment'].value_counts().sort_values(ascending=True),values='Sentiment', names=sentLabels, color_discrete_sequence=['red','blue','green'])

    return pieGraph

@app.callback(Output('update-graph1Mer', 'figure'),
            Input('interval-1Mer', 'n_intervals'),)


def pieGraphMer(n):
    dfMer = pd.read_csv('outputLynch.csv')
    pieGraph = px.pie(dfMer['Sentiment'].value_counts().sort_values(ascending=True),values='Sentiment', names=sentLabelsMer, color_discrete_sequence=['red','blue','green'])
    return pieGraph


@app.callback(Output('top-cards', 'children'),
            Input('interval-cards', 'n_intervals'),)


def update_cards(n):
    return generate_cards()


@app.callback(Output('top-cardsMer', 'children'),
            Input('interval-cardsMer', 'n_intervals'),)


def update_cardsMer(n):
    return generate_cardsMer()




@app.callback(Output('table-graph', 'figure'),
            Input('interval-table', 'n_intervals'),)


def update_table(n):
    df = pd.read_csv('outputBank.csv')    
    generate_table(df, max_rows=20)
    fig = make_subplots(
        rows=1, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=[[{"type": "table"}]]
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Service", "Tweet", "Timestamp", "Location"],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=[df[k].tolist() for k in df.columns[1:]],
                align = "left")
        ),
        row=1, col=1
    )
    fig.update_layout(
        height=1200,
        width=1800,
        showlegend=False,
        title_text="Recent tweets about Bank of America",
    )

    return fig  


@app.callback(Output('table-graphMer', 'figure'),
            Input('interval-tableMer', 'n_intervals'),)


def update_tableMer(n):
    df = pd.read_csv('outputLynch.csv')    
    generate_table(df, max_rows=20)
    fig = make_subplots(
        rows=1, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=[[{"type": "table"}]]
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Service", "Tweet", "Timestamp", "Location"],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=[df[k].tolist() for k in df.columns[1:]],
                align = "left")
        ),
        row=1, col=1
    )
    fig.update_layout(
        height=1200,
        width=1800,
        showlegend=False,
        title_text="Recent tweets about Merrill Lynch",
    )

    return fig  


def addContent(sc):
    print("Refreshing with new content..." + "[]")
    getTweets('Bank of America')
    getTweets('Merrill Lynch')
    sentimentCreation()
    sc.enter(10, 1, addContent, (sc,))

def parallel(*functions):
    processes = []
    for function in functions:
        p = Process(target=function)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()



def secThread():
    print("running w/ new content")
    getTweets('Bank of America')
    getTweets('Merrill Lynch')
    sentimentCreation()
    s.enter(10, 1, addContent, (s,))
    s.run()

def firstThread():
    app.run_server(host= '0.0.0.0',debug=False)

if __name__ == "__main__":
    # first_thread = threading.Thread(target=firstThread())
    # second_thread = threading.Thread(target=secThread())
    # first_thread.start()
    # second_thread.start()
    parallel(secThread, firstThread)

