import re

import nltk
from matplotlib import pyplot as plt
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('averaged_perceptron_tagger_eng')

STOPWORDS = set(stopwords.words("english"))


def get_wordnet_pos(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-z\s]", "", text.lower()).strip()
    tokens = word_tokenize(text)
    return " ".join([word for word in tokens if word not in STOPWORDS])


def plot_word_cloud(text: str, sentiment: str) -> None:
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_text(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(f"Word Cloud for {sentiment.capitalize()} Reviews", fontsize=16)
    plt.axis("off")
    plt.show()


def lemmatize_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    print(pos_tags)
    return " ".join([lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags])
