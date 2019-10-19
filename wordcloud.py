from wordcloud import WordCloud
import matplotlib.pyplot as plt


def generate_wordcloud(text_list):
    text = ' '.join(text_list)
    # Generate a word cloud image
    wordcloud = WordCloud().generate(text)

    wordcloud = WordCloud(max_font_size=60).generate(text)
    plt.figure(figsize=(20,40))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()