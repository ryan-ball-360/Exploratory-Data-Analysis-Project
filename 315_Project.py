#Ryan Ball, 11686775
#Link to dataset used: https://www.kaggle.com/datasets/datasnaek/youtube/code?resource=download
from matplotlib.dviread import Text
import numpy as np
import pandas as pd

from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import string
from string import digits
import re
from scipy.stats import pearsonr



comments = pd.read_csv(r'UScomments.csv', error_bad_lines = False)
videos = pd.read_csv(r'USvideos.csv', error_bad_lines = False)

comments.head()


#clean the data
#all data cleaning steps can be accredited to Sachin Jakasaniya in his EDA project using the same dataset from Kaggle (data cleaning methods was the only step I used from this source). A link to his project is availabe in the acknowledgements section of my report
#1. change text to lower case
comments["comment_text"] = comments["comment_text"].str.lower()
comments.head()


#2 remove puntucation
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

comments['comment_text'] = comments['comment_text'].fillna('')
comments["comment_text"] = comments["comment_text"].map(lambda text: remove_punctuation(text))
comments.head()

#3 remove digits
def remove_digits(text):
    return text.translate(str.maketrans('', '', digits))

comments["comment_text"] = comments["comment_text"].apply(lambda text: remove_digits(text))
comments.head()

#4 remove emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

comments["comment_text"] = comments["comment_text"].apply(lambda text:  remove_emoji(text))
comments.head()

#4 remove short words
comments['comment_text'] = comments['comment_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
comments['comment_text'] = comments['comment_text'].str.replace("[^a-zA-Z#]", " ")
comments.head()


#use the Textblob sentiment analysis to determine comments as positive, negative, or neutral
#this is the most run-time intensive part of the program
sentiment = [TextBlob(comment).sentiment.polarity if isinstance(comment, str) else 0 for comment in comments.comment_text.values]

comments['sentiment']=sentiment
aggregated_sentiment = comments.groupby('video_id')['sentiment'].agg('mean').reset_index(name='sentiment')

videos['sentiment'] = aggregated_sentiment['sentiment']


#aggregate results
aggregated_videos = videos.groupby('video_id').agg({'views' : 'mean', 'sentiment':'mean', 'likes' : 'mean'})
aggregated_videos.head()

#graph results
sns.regplot(data = aggregated_videos, x = 'views', y = 'sentiment').set(title='Correlation Between Sentiment and Views')
plt.show()

# remove NaN values from input arrays
views = aggregated_videos['views'].values
sentiment = aggregated_videos['sentiment'].values
mask = ~np.isnan(views) & ~np.isnan(sentiment)
views = views[mask]
sentiment = sentiment[mask]


# compute Pearson correlation coefficient and p-value
pearson_coef1, p_value1 = pearsonr(views, sentiment)

#dislikes/likes ratio is used rather than likes/dislikes to avoid division 
def calc_ratio(row):
    return row['likes'] / row['dislikes']

# create new dataframe with non-zero likes and dislikes
videos_with_likes_and_dislikes = videos[(videos['likes'] != 0) & (videos['dislikes'] != 0)]

# calculate like to dislike ratio for non-zero entries
videos_with_likes_and_dislikes['like_to_dislike_ratio'] = videos_with_likes_and_dislikes.apply(calc_ratio, axis=1)

# create new dataframe with video_id, views, and like_to_dislike_ratio columns
like_ratio_frame = videos_with_likes_and_dislikes[['video_id', 'views', 'like_to_dislike_ratio']].copy()

sns.regplot(data = like_ratio_frame, x = 'views', y = 'like_to_dislike_ratio').set(title='Correlation Between Like/Dislike Ratio and Views')
plt.show()

pearson_coef2, p_value2 = pearsonr(like_ratio_frame['views'], like_ratio_frame['like_to_dislike_ratio'])

#second dataframe is inconsequential. I made it to test if there was an issue with the first dataframe, but learned my issue was with p-values and never saw the need to change it
videos2 = pd.read_csv(r'USvideos.csv', error_bad_lines = False)
pearson_coef3, p_value3 = pearsonr(videos2['views'], videos2['likes'])

sns.regplot(data = videos2, x = 'views', y = 'likes').set(title='Correlation Between Likes and Views')
plt.show()

pearson_coef4, p_value4 = pearsonr(videos2['views'], videos2['dislikes'])
sns.regplot(data = videos2, x = 'views', y = 'dislikes').set(title='Correlation Between Dislikes and Views')
plt.show()

pearson_coef5, p_value5 = pearsonr(videos2['views'], videos2['comment_total'])
sns.regplot(data = videos2, x = 'views', y = 'comment_total').set(title='Correlation Between Total Comments and Views')
plt.show()

category_dict = {
    1:'Film & Animation',
    2:'Autos & Vehicles',
    10: 'Music',
    15: 'Pets & Animals',
    17: 'Sports',
    18: 'Short Movies',
    19: 'Travel & Events',
    20: 'Gaming',
    21: 'Videoblogging',
    22: 'People & Blogs',
    23: 'Comedy',
    24: 'Entertainment',
    25: 'News & Politics',
    26: 'Howto & Style',
    27: 'Education',
    28: 'Science & Technology',
    29: 'Nonprofits & Activism',
    30: 'Movies',
    31: 'Anime/Animation',
    32: 'Action/Adventure',
    33: 'Classics',
    34: 'Comedy',
    35: 'Documentary',
    36: 'Drama',
    37: 'Family',
    38: 'Foreign',
    39: 'Horror',
    40: 'Sci-Fi/Fantasy',
    41: 'Thriller',
    42: 'Shorts',
    43: 'Shows',
    44: 'Trailers'
}

# Group by category_id and sum the views
videos = videos.groupby('category_id')['views'].sum()
# Rename index using category_dict
videos.index = videos.index.map(category_dict)
# Sort the categories by views in descending order
videos_sorted = videos.sort_values(ascending=False)
# Plot the bar graph
category_graph = videos_sorted.plot(kind='bar', title='Video Categories by Views', figsize=(7, 3))
category_graph.set_xlabel('Category')
category_graph.set_ylabel('Views')
plt.show()

# Create a dictionary with the coefficients and their corresponding labels
correlations = {
    'Views-Sentiment': pearson_coef1,
    'Views-Like/Dislike Ratio': pearson_coef2,
    'Views-Likes': pearson_coef3,
    'Views-Dislikes': pearson_coef4,
    'Views-Total Comments': pearson_coef5
}

print(f"Correlation Between Views and Sentiment: Pearson Coefficient = {pearson_coef1}, p-value = {p_value1}")
print(f"Correlation Between Views and Likes: Pearson Coefficient = {pearson_coef3}, p-value = {p_value3}")
print(f"Correlation Between Views and Likes: Pearson Coefficient = {pearson_coef3}, p-value = {p_value3}")
print(f"Correlation Between Views and Dislikes: Pearson Coefficient = {pearson_coef4}, p-value = {p_value4}")
print(f"Correlation Between Views and Total Comments: Pearson Coefficient = {pearson_coef5}, p-value = {p_value5}")


# Sort the dictionary by values in descending order
sorted_correlations = {k: v for k, v in sorted(correlations.items(), key=lambda item: item[1], reverse=True)}

# Print the sorted correlations
for label, coef in sorted_correlations.items():
    print(f'{label}: {coef}')
#for this program I simply used the run and debug, and the run code in interactive window features from VS code 