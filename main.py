import string

import nltk
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import csv
import glob


class BM25(object):
    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False, ngram_range=(1,2))
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1

def preProcess(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words())
    stop_dict = {s: 1 for s in stopwords_set}
    s = [w for w in s if w not in stop_dict]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s

def tfidf_search(csvFile):
    query = input("Enter lyric: ")
    score = input("Enter score name: ")
    df = pd.read_csv(csvFile)
    df.drop_duplicates(subset="Lyric",
                         keep='last', inplace=True)
    vectorizer = TfidfVectorizer()
    if score == "tf":
        lyric = df['Lyric']
        vectorizer = CountVectorizer(preprocessor=preProcess)
        vectorizer.fit_transform(lyric)
        results = vectorizer.transform([query])
        rank = 0
        for i in results.argsort()[-10:][::-1]:
            rank += 1
            print("Rank: ", rank, " Artist: ", df.iloc[i, 0], " Lyric: ", df.iloc[i, 3])
    if score == "tf-idf":
        X = vectorizer.fit_transform(df['Lyric'].apply(lambda x: np.str(x)))
        query_vectorizer = vectorizer.transform([query])
        results = cosine_similarity(X, query_vectorizer).reshape((-1,))
        rank = 0
        for i in results.argsort()[-10:][::-1]:
            rank+=1
            print("Rank: ", rank , " Artist: ", df.iloc[i, 0], " Lyric: ", df.iloc[i, 3])
    if score == "bm25":
            bm25 = BM25()
            bm25.fit(df["Lyric"].astype('U'))
            results = bm25.transform(query, df["Lyric"].astype('U'))
            rank = 0
            for i in results.argsort()[-10:][::-1]:
                rank += 1
                print("Rank: ", rank, " Artist: ", df.iloc[i, 0], " Lyric: ", df.iloc[i, 3])
    else:
        print("No score name existed")

def search_song_name(csvFile):
    csv_file = csv.reader(open(csvFile, 'r', encoding='utf-8'))
    song_name = input("Enter song name: ")

    with open('src/resource/lyrics2.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Artists', 'Lyrics'])
        for row in csv_file:
            if song_name == row[1]:
                writer.writerow([row[0], row[3]])
                # print("Song name: " + row[1] + "\n"
                #     + "Lyric: " + row[3])

    data = pd.read_csv('src/resource/lyrics2.csv')
    data.drop_duplicates(subset="Lyrics",
                         keep='last', inplace=True)

    print(data)

def search_artist(csvFile):

    csv_file = csv.reader(open(csvFile, 'r', encoding='utf-8'))
    artist_name = input("Enter artist name: ")

    artist_name = "/" + artist_name.replace(" ", "-").lower() + "/"
    print(artist_name)
    with open('src/resource/artist.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Artists', 'SName'])
        for row in csv_file:
            if artist_name == row[0]:
                writer.writerow([row[0], row[1]])
                # print("Artist: " + row[0] + "\n"
                #     + "SName: " + row[3])

    data = pd.read_csv('src/resource/artist.csv')
    data.drop_duplicates(subset="SName",
                         keep='last', inplace=True)

    print(data)

li = []
path = 'src/resource/'
all_files = glob.glob(path + "*.csv")
def read_multiple_csv_file():

    li_mapper = map(lambda filename: pd.read_csv(
        filename, index_col=None, header=0), all_files)
    li_2 = list(li_mapper)

    df = pd.concat(li_2, axis=0, ignore_index=True)
    print(df.head(10))

def create_new_csv_file(csvFile):
    csv_file = csv.reader(open(csvFile, 'r', encoding='utf-8'))

    with open('src/resource/artist.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Artists', 'SName'])
        for row in csv_file:
            writer.writerow([row[0], row[1]])
            # print("Artist: " + row[0] + "\n"
            #     + "SName: " + row[3])

    data = pd.read_csv('src/resource/artist.csv')
    data.drop_duplicates(subset="SName",
                         keep='last', inplace=True)

    print(data)

if __name__ == '__main__':
    csvFile = 'src/resource/lyrics-data.csv'

    print("Enter 1 to choose the tf, tf-idf, bm25 ranking")
    print("Enter 2 to choose search song name")
    print("Enter 3 to choose search artist name")
    choice = int(input("Enter your choice:\n "))
    if choice == 1:
        tfidf_search(csvFile)
    elif choice == 2:
        search_song_name(csvFile)
    elif choice == 3:
        search_artist(csvFile)
    else:
        print("Invalid input")