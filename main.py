# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor


# Import dataset from CSV
data = pd.read_csv("Instagram data.csv", encoding='latin1')

# Plot distribution of Impression from Home page
plt.figure(figsize=(8,6))
plt.style.use('seaborn-v0_8-pastel')
plt.title("Distribution of Impressions from Home")
sns.distplot(data['From Home'])
plt.show()

# Plot distribution of Impressions from Hashtags
plt.figure(figsize=(8,6))
plt.title("Distribution of Impressions from Hashtags")
sns.distplot(data['From Hashtags'])
plt.show()

# Plot distribution of Impressions from Explore page
plt.figure(figsize=(8,6))
plt.title("Distribution of Impressions from Explore")
sns.distplot(data['From Explore'])
plt.show()

# Plot comparison of Impressions from Home, Hashtags and Explore
home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home', 'From Hashtags', 'From Explore','From Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels, title='Impressions on Instagram Posts from Sources', hole=0.3)

fig.write_html('output_file_name.html', auto_open=True)

# Create view to analyse content of Captions
text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('seaborn-v0_8-pastel')
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Create view to analyse content of Hashtags
text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('seaborn-v0_8-pastel')
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Plot relationship between Impressions and Likes
figure = px.scatter(data_frame=data,
                    x="Impressions",
                    y="Likes",
                    size="Likes",
                    trendline="ols",
                    title="Relationship Between Likes and Impressions"
                    )
figure.write_html('output_file_name.html', auto_open=True)

# Plot relationship between Impressions and Comments
figure = px.scatter(data_frame=data,
                    x="Impressions",
                    y="Comments",
                    size="Comments",
                    trendline="ols",
                    title="Relationship Between Comments and Impressions"
                    )
figure.write_html('output_file_name.html', auto_open=True)

# Plot relationship between Impressions and Shares
figure = px.scatter(data_frame=data,
                    x="Impressions",
                    y="Shares",
                    size="Shares",
                    trendline="ols",
                    title="Relationship Between Comments and Shares"
                    )
figure.write_html('output_file_name.html', auto_open=True)

# Plot relationship between Impressions and Saves
figure = px.scatter(data_frame=data,
                    x="Impressions",
                    y="Saves",
                    size="Saves",
                    trendline="ols",
                    title="Relationship Between Comments and Saves"
                    )
figure.write_html('output_file_name.html', auto_open=True)

# Analyse correlation between Impressions and other metrics
correlation = data.corr()
print(correlation["Impressions"].sort_values(ascending=False))

# Analyse conversion rate
conversion_rate = (data["Follows"].sum()/data["Profile Visits"].sum())*100
print(conversion_rate)

# Plot relationship between Profile visits and Follows
figure = px.scatter(data_frame=data,
                    x="Profile Visits",
                    y="Follows",
                    size="Follows",
                    trendline="ols",
                    title="Relationship Between Profile Visits and Follows"
                    )
figure.write_html('output_file_name.html', auto_open=True)

# Model to predict Instagram Reach of a post
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
# score = model.score(xtest, ytest)
# print(score)

# Predict reach of post. Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
features = np.array([[400, 300, 15, 20, 100, 60]])
print(model.predict(features))

