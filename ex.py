from NewsClassifier import ClassifierModels
from NewsClassifier.modelTransformers import Cleaner, CountVectorizerWithStemming, SetBias, RemoveNan
from newsapi import NewsApiClient
import numpy as np
from datetime import date, timedelta
#a = ClassifierModels.FinFilter().fit_predict()

news = NewsApiClient(api_key='8e500c992adf43d680f652f336311b61')
# a = news.get_everything(q="Apple Inc.",
#                         from_param=(date.today() - timedelta(days=1)).strftime('%Y-%m-%d'),
#                         language='en',
#                         sort_by='relevancy',
#                         page_size= 2,
#                         page=1)['articles']
#query_list = [[article['source']['name'], article['title'], article['description'], article['content']] for article in a ]
query_list = np.array([['Mashable',
        'Apple also uses humans to listen to some Siri recordings',
        'News alert: your Siri voice recordings may not be entirely private. According to The Guardian, Apple hires contractors to listen to Siri recordings in order to improve the accuracy and quality of the voice assistant. These contractors "regularly hear confiden…',
        'News alert: your Siri voice recordings may not be entirely private.\r\nAccording to The Guardian, Apple hires contractors to listen to Siri recordings in order to improve the accuracy and quality of the voice assistant.\r\nThese contractors "regularly hear confid… [+3422 chars]'],
       ['CNN',
        "Apple Mac Pro parts won't be exempt from China tariffs, Trump says",
        "Apple would like to avoid paying additional tariffs if the Trump administration escalates its trade war with China. President Donald Trump has thrown cold water on Apple's hopes.",
        'Apple will not be given Tariff waiver, or relief, for Mac Pro parts that are made in China. Make them in the USA, no Tariffs!\r\n— Donald J. Trump (@realDonaldTrump) July 26, 2019'],
       ['Engadget',
        "The Morning After: Google's Pixel sales doubled thanks to the 3a",
        "Hey, good morning! You look fabulous. This morning, we've got an update on SpaceX's latest rocket launches and the details on Apple's billion-dollar acquisition. Also, we're looking for your impressions on Google's Nest Hub, and it seems hoverboarding across …",
        "Check out the engine cam video.SpaceX sends its 'Starhopper' on an untethered test flight\r\nAfter completing zero of two launch attempts on Wednesday, SpaceX went two for two on Thursday. Its Dragon spacecraft is making a third trip to the ISS, while its Starh… [+3068 chars]"]])
val = ClassifierModels.Classifier().fit_predict(query_list).toarray()
print(np.isnan(np.sum(val)))