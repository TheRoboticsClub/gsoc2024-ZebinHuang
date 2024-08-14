import os
from collections import Counter
import ast
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def analysis(file_path, pie_chart_filename=None, word_cloud_filename=None, output_dir='data_parsing/results'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    # Set default filenames if not provided
    pie_chart_filename = pie_chart_filename or f'{dataset_name}_action_distribution.png'
    word_cloud_filename = word_cloud_filename or f'{dataset_name}_instructions_wordcloud.png'

    # Read the dataset and extract action labels and their counts
    data = pd.read_csv(file_path)
    action_counts = Counter([action for sublist in data['action'] for action in ast.literal_eval(sublist)])
    labels = action_counts.keys()
    sizes = action_counts.values()

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    ax.axis('equal')

    # Plot the pie chart with a white circle at the centre
    centre_circle = plt.Circle((-0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    pie_chart_fig = fig
    plt.legend(wedges, labels, title="Actions", loc="center left", bbox_to_anchor=(0.05, 0, 0.5, 1))
    plt.title('Distribution of Action Labels')

    pie_chart_path = os.path.join(output_dir, pie_chart_filename)
    plt.savefig(pie_chart_path)
    plt.show()

    # Generate word cloud of instructions
    text = ' '.join(data['instruction'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    wordcloud_fig, wordcloud_ax = plt.subplots(figsize=(10, 5))
    wordcloud_ax.imshow(wordcloud, interpolation='bilinear')
    wordcloud_ax.axis('off')
    plt.title('Word Cloud of Instructions')

    word_cloud_path = os.path.join(output_dir, word_cloud_filename)
    plt.savefig(word_cloud_path)
    plt.show()
    return wordcloud_fig, pie_chart_fig
