import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 1. Defining Problem Statement and Analyzing basic metrics

data=pd.read_csv("netflix.csv")
data.head()
data.tail()
data.columns

# 2.Observations on the shape of data, data types of all the attributes, conversion of categorical attributes to 'category' (If required), missing value detection, statistical summary

data.shape              # No. of rows and columns
data.info()             #data types of all attributes
data.describe(include="all")
data.nunique()
data.duplicated().sum()

# Handling Missing Values

data.isnull().sum().sort_values(ascending=False)        # No. of null values in each series 
round(data.isnull().sum()/data.shape[0]*100,2).sort_values(ascending=False)

# For series - duration, rating and date added the % of missing values is very low. Hence dropping those rows.

data.dropna(subset=["duration","rating","date_added"],axis=0,inplace=True)

# For series (director, cast and country) significant amount of data is missing. So, replacing it with appropriate value.

data["country"].replace(np.NaN,"Unknown",inplace=True)
data["cast"].replace(np.NaN,"Unknown",inplace=True)
data["director"].replace(np.NaN,"Unknown",inplace=True)

round(data.isnull().sum()/data.shape[0]*100,2).sort_values(ascending=False)

# Converting datatypes

data["date_added"] = pd.to_datetime(data["date_added"],format="mixed")
data["day_added"] = data["date_added"].dt.day
data["year_added"] = data["date_added"].dt.year
data["month_added"]=data["date_added"].dt.month
data["year_added"].astype(int)
data["day_added"].astype(int)

data["duration"]=data["duration"].apply(lambda x: str(x).split(" ")[0])
data["duration"]=data["duration"].astype(int)

data.describe()           # Statistical summary

# Given data has 8807 rows and 12 columns In raw form all the columns has object data type except for release_year which has int64 dtype There is no duplicate row in the dataset

# 3. Non-Graphical Analysis: Value counts and unique attributes

# Comparison of tv shows vs. movies.

data['type'].value_counts()
x=data['type'].value_counts(normalize=True)*100
print(round(x,2))

# Longest and smallest running movie and TV-show

movie_df=data.loc[data["type"]=="Movie"]
tv_df=data.loc[data["type"]=="TV Show"]

#Shortest movie

shortest_movie=movie_df.loc[movie_df["duration"]==np.min(movie_df.duration)]
shortest_movie

# Rating of short running Movies

short_movieR=movie_df.loc[movie_df["duration"]<60]
short_movieR.rating.value_counts().sort_values(ascending=False)

#Longest movie

longest_movie=movie_df.loc[movie_df["duration"]==np.max(movie_df.duration)]
longest_movie

# Rating of long running Movies

longest_movieR=movie_df.loc[movie_df["duration"]>180]
longest_movieR.rating.value_counts().sort_values(ascending=False)

#Shortest tv show

shortest_tv=tv_df.loc[tv_df["duration"]==np.min(tv_df.duration)]
shortest_tv.head()

# Rating of short running tv shows

short_showR=tv_df.loc[tv_df["duration"]==1]
short_showR.rating.value_counts().sort_values(ascending=False)


#Longest TV Show

longest_tv=tv_df.loc[tv_df["duration"]==np.max(tv_df.duration)]
longest_tv

# Rating of long running tv shows

long_showR=tv_df.loc[tv_df["duration"]>=10]
long_showR.rating.value_counts().sort_values(ascending=False)

# Number of movies and TV-Shows added to netflix each year

data.groupby("year_added").ngroups
data.groupby("year_added")["type"].value_counts()
data.groupby(["year_added","month_added"])["type"].value_counts()

# Year in which maximum and minimum movies were added

movie_df.groupby("year_added")["type"].value_counts().sort_values(ascending=False)

# Year-month in which maximum and minimum movies were added

movie_df.groupby(["year_added","month_added"])["type"].value_counts().sort_values(ascending=False)

# Year in which maximum and minimum tv shows were added

tv_df.groupby("year_added")["type"].value_counts().sort_values(ascending=False)

# Year-month in which maximum and minimum tv shows were added

tv_df.groupby(["year_added","month_added"])["type"].value_counts().sort_values(ascending=False)


#Analysis on rating

data["rating"].nunique()

#Count of different ratings

data.rating.value_counts().sort_values(ascending=False)

#Movies and tv show rating count

data.groupby("rating")["type"].value_counts()

#Movies of max and min rating type

movie_df.rating.value_counts().sort_values(ascending=False)


#TV shows of max and min rating type

tv_df.rating.value_counts().sort_values(ascending=False)

#Year wise rating count

data.groupby("year_added")['rating'].value_counts()


# Preprocessing of data

cast_list=data['cast'].apply(lambda x: str(x).split(', ')).tolist()
df_cast=pd.DataFrame(cast_list,index=data['title'])
df_cast=df_cast.stack()
df_cast=pd.DataFrame(df_cast)
df_cast.reset_index(inplace=True)
df_cast=df_cast[['title',0]]
df_cast.columns=['title','cast']
df_cast.head()

director_list=data['director'].apply(lambda x: str(x).split(', ')).tolist()
df_director=pd.DataFrame(director_list,index=data['title'])
df_director=df_director.stack()
df_director=pd.DataFrame(df_director)
df_director.reset_index(inplace=True)
df_director=df_director[['title',0]]
df_director.columns=['title','director']
df_director.head()

country_list=data['country'].apply(lambda x: str(x).split(', ')).tolist()
df_country=pd.DataFrame(country_list,index=data['title'])
df_country=df_country.stack()
df_country=pd.DataFrame(df_country)
df_country.reset_index(inplace=True)
df_country=df_country[['title',0]]
df_country.columns=['title','country']
df_country.head()

genre_list=data['listed_in'].apply(lambda x: str(x).split(', ')).tolist()
df_genre=pd.DataFrame(genre_list,index=data['title'])
df_genre=df_genre.stack()
df_genre=pd.DataFrame(df_genre)
df_genre.reset_index(inplace=True)
df_genre=df_genre[['title',0]]
df_genre.columns=['title','genre']
df_genre.head()

df_l=df_cast.merge(df_director,left_on="title",right_on="title",how="inner")
df_l.head()

df_r=df_country.merge(df_genre,left_on="title",right_on="title",how="inner")
df_r.head()

dff=df_l.merge(df_r,left_on="title",right_on="title",how="inner")

df.drop(["director_x","cast_x","listed_in","country_x"],axis=1,inplace=True)

df.rename(columns={"cast_y":"cast",
                    "director_y":"director",
                    "country_y":"country"},inplace=True)


# Actor featured in most movies and tv shows

x=df.loc[df["cast"]!="Unknown"]
x.groupby("cast")["title"].nunique().sort_values(ascending=False)

df_movie=df.loc[df["type"]=="Movie"]
df_tv=df.loc[df["type"]=="TV Show"]

# Actor featured in most movies

y=df_movie.loc[df_movie["cast"]!="Unknown"]
y.groupby("cast")["title"].nunique().sort_values(ascending=False)

# Actor featured in most tv-shows

z=df_tv.loc[df_tv["cast"]!="Unknown"]
z.groupby("cast")["title"].nunique().sort_values(ascending=False)

#Most popular director

a=df.loc[df["director"]!="Unknown"]
a.groupby("director")["title"].nunique().sort_values(ascending=False)

# Most popular genre

df.groupby("genre")["title"].nunique().sort_values(ascending=False)

# Countries with maximum titles

df.groupby("country")["title"].nunique().sort_values(ascending=False)


# Visual Analysis - Univariate, Bivariate after pre-processing of the data

data['type'].value_counts().plot(kind='pie',autopct="%.2f%%")
plt.title("Movies vs TV-Shows")

plt.figure(figsize=(6,4))
sns.histplot(data["rating"], kde=True)
plt.xticks(rotation=60)
plt.title("Count of Ratings")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='year_added', hue='type', data=data,width=1)
plt.xticks(rotation=60,fontsize=8)
plt.title("Movies & TV-Shows added each year")

plt.figure(figsize=(6,4))
sns.kdeplot(x="month_added",data=data,hue="type")

plt.figure(figsize=(6,4))
sns.countplot(x='rating',hue='type',data=data)
plt.xticks(rotation=60)
plt.title('Relation between Type and Rating')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(x='release_year',hue='type',data=data)
plt.xticks(rotation=60)
plt.title("Movies/Shows vs Released Year")
plt.show()

# Most popular cast

x=df.loc[df["cast"]!="Unknown"]
b=x.groupby("cast")["title"].nunique().sort_values(ascending=False)
b=pd.DataFrame(b)
b.reset_index(inplace=True)
b.rename(columns={"title":"count"},inplace=True)
d_c=b.head(10)
plt.figure(figsize=(6,4))
Y1=d_c["count"]
X1=d_c["cast"]
plt.barh(X1,Y1)
plt.title("Top 10 Actors")
plt.ylabel("Actor")
plt.xlabel("Count")
plt.show()

#Most popular genre

g=df.groupby("genre")["title"].nunique().sort_values(ascending=False)
g=pd.DataFrame(g)
g.reset_index(inplace=True)
g.rename(columns={"title":"count"},inplace=True)
d_g=g.head(10)
plt.figure(figsize=(6,4))
Yg=d_g["count"]
Xg=d_g["genre"]
plt.barh(Xg,Yg)
plt.title("Top 10 Genres")
plt.ylabel("Genre")
plt.xlabel("Count")
plt.show()


# For categorical variable(s): Boxplot

plt.figure(figsize=(6,4))
sns.boxplot(data=data,x="year_added")
plt.title("Spread of Content Added to Netflix")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=data,x="release_year")
plt.title("Spread of Release Date")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=data,x="month_added")
plt.title("Spread of Months")
plt.show()

#For correlation: Heatmaps, Pairplots

sns.pairplot(data=data,hue="type")

sns.heatmap(data[["year_added","release_year"]])

# Outliers check

plt.figure(figsize=(6,4))
sns.boxplot(data=data,x="day_added")
plt.title("Spread of Months")
plt.show()