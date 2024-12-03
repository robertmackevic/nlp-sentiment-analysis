import re

import spacy
from matplotlib import pyplot as plt
from wordcloud import WordCloud

NLP = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-z\s]", "", text.lower()).strip()
    return " ".join([word for word in text.split() if word not in NLP.Defaults.stop_words])


def lemmatize_text(text: str) -> str:
    text = re.sub(r"[^a-z\s']", "", text.lower()).strip()
    return " ".join([token.lemma_ for token in NLP(text) if not token.is_stop])


def plot_word_cloud(text: str, sentiment: str) -> None:
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_text(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(f"Word Cloud for {sentiment.capitalize()} Reviews", fontsize=16)
    plt.axis("off")
    plt.show()
