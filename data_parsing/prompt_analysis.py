import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

output_dir = 'results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_path = 'dataset_1000.csv'
dataset_name = os.path.splitext(os.path.basename(file_path))[0]

data = pd.read_csv(file_path)
data['action'] = data['action'].apply(ast.literal_eval)
action_counts = Counter([action for sublist in data['action'] for action in sublist])
labels = action_counts.keys()
sizes = action_counts.values()

# Create the pie chart
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
ax.axis('equal')

centre_circle = plt.Circle((-0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

plt.legend(wedges, labels, title="Actions", loc="center left", bbox_to_anchor=(0.05, 0, 0.5, 1))
plt.title('Distribution of Action Labels')

pie_chart_filename = os.path.join(output_dir, f'{dataset_name}_action_distribution.png')
plt.savefig(pie_chart_filename)
plt.show()

text = ' '.join(data['instruction'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Instructions')

word_cloud_filename = os.path.join(output_dir, f'{dataset_name}_instructions_wordcloud.png')
plt.savefig(word_cloud_filename)
plt.show()
