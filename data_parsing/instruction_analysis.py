import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Generate action distribution pie chart and word cloud from dataset.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the CSV dataset file.')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save the output files.')
    parser.add_argument('--pie_chart_filename', type=str, help='Filename for the pie chart image.')
    parser.add_argument('--word_cloud_filename', type=str, help='Filename for the word cloud image.')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset_name = os.path.splitext(os.path.basename(args.file_path))[0]

    # Set default filenames if not provided
    pie_chart_filename = args.pie_chart_filename or f'{dataset_name}_action_distribution.png'
    word_cloud_filename = args.word_cloud_filename or f'{dataset_name}_instructions_wordcloud.png'

    data = pd.read_csv(args.file_path)
    action_counts = Counter([action for sublist in data['action'] for action in ast.literal_eval(sublist)])
    labels = action_counts.keys()
    sizes = action_counts.values()

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    ax.axis('equal')

    centre_circle = plt.Circle((-0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    plt.legend(wedges, labels, title="Actions", loc="center left", bbox_to_anchor=(0.05, 0, 0.5, 1))
    plt.title('Distribution of Action Labels')

    pie_chart_path = os.path.join(args.output_dir, pie_chart_filename)
    plt.savefig(pie_chart_path)
    plt.show()

    text = ' '.join(data['instruction'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Instructions')

    word_cloud_path = os.path.join(args.output_dir, word_cloud_filename)
    plt.savefig(word_cloud_path)
    plt.show()


if __name__ == '__main__':
    main()
