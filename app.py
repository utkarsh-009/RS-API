#Importing Dependencies

from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dependencies to get movie poster
from bs4 import BeautifulSoup
import requests,io
import PIL.Image
from urllib.request import urlopen
import streamlit as st



# Data Collection and Pre-Processing
url = "https://drive.google.com/file/d/1wmHJ4aRSddl6uUl95a5T1JibDNjaL0Kn/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
movies_data = pd.read_csv(path)


# selecting the relevant features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']

# replacing the null valuess with null string
for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')

# combining all the 5 selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

# converting the text data to feature vectors
vectorizer = TfidfVectorizer()

# converting combined_features into feature vectors and returning it into matrix form
feature_vectors = vectorizer.fit_transform(combined_features)

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)

def movie_poster_fetcher(imdb_link):
    ## Display Movie Poster
    url_data = requests.get(imdb_link).text
    s_data = BeautifulSoup(url_data, 'html.parser')
    imdb_dp = s_data.find("meta", property="og:image")
    movie_poster_link = str(imdb_dp.attrs['content'])
    return movie_poster_link
    # u = urlopen(movie_poster_link)
    # raw_data = u.read()
    # image = PIL.Image.open(io.BytesIO(raw_data))
    # image = image.resize((158, 301), )
    # st.image(image, use_column_width=False)s

# initial recommendation
inital_mlist = [
  {
    "description": "Stuart, an adorable white mouse, still lives happily with his adoptive family, the Littles, on the east side of Manhattan's Central Park. More crazy mouse adventures are in store as Stuart, his human brother, George, and their mischievous cat, Snowbell, set out to rescue a friend.",
    "homepage": "http://www.imdb.com/title/tt0243585/?ref_=fn_tt_tt_1",
    "id": 0,
    "image_url": "https://m.media-amazon.com/images/M/MV5BMTYyNDg0Njc2Nl5BMl5BanBnXkFtZTYwMDc3NzQ3._V1_FMjpg_UX1000_.jpg",
    "movie_name": "Stuart Little 2"
  },
  {
    "description": "Bound by a shared destiny, a bright, optimistic teen bursting with scientific curiosity and a former boy-genius inventor jaded by disillusionment embark on a danger-filled mission to unearth the secrets of an enigmatic place somewhere in time and space that exists in their collective memory as \"Tomorrowland.\"",
    "homepage": "http://www.imdb.com/title/tt1964418/?ref_=fn_tt_tt_1",
    "id": 1,
    "image_url": "https://m.media-amazon.com/images/M/MV5BMTQ4OTgzNTkwNF5BMl5BanBnXkFtZTgwMzI3MDE3NDE@._V1_FMjpg_UX1000_.jpg",
    "movie_name": "Tomorrowland"
  },
  {
    "description": "A young boy and his dog, who happens to have a genius-level IQ, spring into action when their time-travel machine is stolen and moments in history begin to be changed.",
    "homepage": "http://www.imdb.com/title/tt0864835/?ref_=fn_tt_tt_1",
    "id": 2,
    "image_url": "https://m.media-amazon.com/images/M/MV5BMTkxMzM0NzcwN15BMl5BanBnXkFtZTgwNzk1MjMzMTE@._V1_FMjpg_UX1000_.jpg",
    "movie_name": "Mr. Peabody & Sherman"
  },
  {
    "description": "When Susan Murphy is unwittingly clobbered by a meteor full of outer space gunk on her wedding day, she mysteriously grows to 49-feet-11-inches. The military jumps into action and captures Susan, secreting her away to a covert government compound. She is renamed Ginormica and placed in confinement with a ragtag group of Monsters...",
    "homepage": "http://www.imdb.com/title/tt0892782/?ref_=fn_tt_tt_1",
    "id": 3,
    "image_url": "https://m.media-amazon.com/images/M/MV5BMTY0OTQ3MzE3MV5BMl5BanBnXkFtZTcwMDQyMzMzMg@@._V1_FMjpg_UX1000_.jpg",
    "movie_name": "Monsters vs Aliens"
  },
  {
    "description": "In RoboCop, the year is 2028 and multinational conglomerate OmniCorp is at the center of robot technology.  Overseas, their drones have been used by the military for years, but have been forbidden for law enforcement in America.  Now OmniCorp wants to bring their controversial technology to the home front, and they see a golden opportunity to do it.  When Alex Murphy – a loving husband, father and good cop doing his best to stem the tide of crime and corruption in Detroit – is critically injured, OmniCorp sees their chance to build a part-man, part-robot police officer.  OmniCorp envisions a RoboCop in every city and even more billions for their shareholders, but they never counted on one thing: there is still a man inside the machine.",
    "homepage": "http://www.imdb.com/title/tt1234721/?ref_=fn_tt_tt_1",
    "id": 4,
    "image_url": "https://m.media-amazon.com/images/M/MV5BMjAyOTUzMTcxN15BMl5BanBnXkFtZTgwMjkyOTc1MDE@._V1_FMjpg_UX1000_.jpg",
    "movie_name": "RoboCop"
  },
  {
    "description": "When Earth is taken over by the overly-confident Boov, an alien race in search of a new place to call home, all humans are promptly relocated, while all Boov get busy reorganizing the planet. But when one resourceful girl, Tip, manages to avoid capture, she finds herself the accidental accomplice of a banished Boov named Oh. The two fugitives realize there’s a lot more at stake than intergalactic relations as they embark on the road trip of a lifetime.",
    "homepage": "http://www.imdb.com/title/tt2224026/?ref_=fn_tt_tt_1",
    "id": 5,
    "image_url": "https://m.media-amazon.com/images/M/MV5BMjExOTQ4MDMyMV5BMl5BanBnXkFtZTgwMTE3NDM2MzE@._V1_FMjpg_UX1000_.jpg",
    "movie_name": "Home"
  },
  {
    "description": "It's a jungle out there for Blu, Jewel and their three kids after they're hurtled from Rio de Janeiro to the wilds of the Amazon. As Blu tries to fit in, he goes beak-to-beak with the vengeful Nigel, and meets the most fearsome adversary of all: his father-in-law.",
    "homepage": "http://www.imdb.com/title/tt2357291/?ref_=fn_tt_tt_1",
    "id": 6,
    "image_url": "https://m.media-amazon.com/images/M/MV5BMTgzMDczMDYzNl5BMl5BanBnXkFtZTgwMzk2MDIwMTE@._V1_FMjpg_UX1000_.jpg",
    "movie_name": "Rio 2"
  },
  {
    "description": "Speed Racer is the tale of a young and brilliant racing driver. When corruption in the racing leagues costs his brother his life, he must team up with the police and the mysterious Racer X to bring an end to the corruption and criminal activities. Inspired by the cartoon series.",
    "homepage": "http://www.imdb.com/title/tt0811080/?ref_=fn_tt_tt_1",
    "id": 7,
    "image_url": "https://m.media-amazon.com/images/M/MV5BMWU4Y2RhYzMtYzIxZC00NmRlLTk0OTctNDg1NTg5Yjk4YjQzXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_FMjpg_UX1000_.jpg",
    "movie_name": "Speed Racer"
  },
  {
    "description": "After the cataclysmic events in New York with The Avengers, Steve Rogers, aka Captain America is living quietly in Washington, D.C. and trying to adjust to the modern world. But when a S.H.I.E.L.D. colleague comes under attack, Steve becomes embroiled in a web of intrigue that threatens to put the world at risk. Joining forces with the Black Widow, Captain America struggles to expose the ever-widening conspiracy while fighting off professional assassins sent to silence him at every turn. When the full scope of the villainous plot is revealed, Captain America and the Black Widow enlist the help of a new ally, the Falcon. However, they soon find themselves up against an unexpected and formidable enemy—the Winter Soldier.",
    "homepage": "http://www.imdb.com/title/tt1843866/?ref_=fn_tt_tt_1",
    "id": 8,
    "image_url": "https://m.media-amazon.com/images/M/MV5BMzA2NDkwODAwM15BMl5BanBnXkFtZTgwODk5MTgzMTE@._V1_FMjpg_UX1000_.jpg",
    "movie_name": "Captain America: The Winter Soldier"
  },
  {
    "description": "84 years later, a 101-year-old woman named Rose DeWitt Bukater tells the story to her granddaughter Lizzy Calvert, Brock Lovett, Lewis Bodine, Bobby Buell and Anatoly Mikailavich on the Keldysh about her life set in April 10th 1912, on a ship called Titanic when young Rose boards the departing ship with the upper-class passengers and her mother, Ruth DeWitt Bukater, and her fiancé, Caledon Hockley. Meanwhile, a drifter and artist named Jack Dawson and his best friend Fabrizio De Rossi win third-class tickets to the ship in a game. And she explains the whole story from departure until the death of Titanic on its first and last voyage April 15th, 1912 at 2:20 in the morning.",
    "homepage": "http://www.imdb.com/title/tt0120338/?ref_=fn_tt_tt_1",
    "id": 9,
    "image_url": "https://m.media-amazon.com/images/M/MV5BMDdmZGU3NDQtY2E5My00ZTliLWIzOTUtMTY4ZGI1YjdiNjk3XkEyXkFqcGdeQXVyNTA4NzY1MzY@._V1_FMjpg_UX1000_.jpg",
    "movie_name": "Titanic"
  },
  {
    "description": "Star race car Lightning McQueen and his pal Mater head overseas to compete in the World Grand Prix race. But the road to the championship becomes rocky as Mater gets caught up in an intriguing adventure of his own: international espionage.",
    "homepage": "http://www.imdb.com/title/tt1216475/?ref_=fn_tt_tt_1",
    "id": 10,
    "image_url": "https://m.media-amazon.com/images/M/MV5BMTUzNTc3MTU3M15BMl5BanBnXkFtZTcwMzIxNTc3NA@@._V1_FMjpg_UX1000_.jpg",
    "movie_name": "Cars 2"
  }
]


app = Flask(__name__)

@app.route("/api/<string:movie_name>")

def movieRs(movie_name):
    # movie_name = str(request.args(['query']))
    list_of_all_titles = movies_data['title'].tolist()
    mlist = []

    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if(len(find_close_match) == 0):
        mlist  = inital_mlist
        return jsonify(mlist)

    
    close_match = find_close_match[0]
    
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    
    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

    i = 0

    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index =  movies_data[movies_data.index == index]['title'].values[0]
        homepage_from_index = movies_data[movies_data.index == index]['homepage'].values[0]
        desc = movies_data[movies_data.index == index]['overview'].values[0]
        if (i <= 10):
            movie_poster_url = movie_poster_fetcher(homepage_from_index)
            mlist.append({'id': i, 'movie_name': title_from_index, 'homepage': homepage_from_index, 'image_url': movie_poster_url, 'description': desc})
            i += 1
    
    return jsonify(mlist)



if __name__ == "__main__":
    app.run(debug=False, host = '0.0.0.0',port = 5000)

