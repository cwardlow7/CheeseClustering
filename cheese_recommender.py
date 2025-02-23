import pandas as pd
import umap as umap
from sklearn.cluster import HDBSCAN
import geopandas
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nbformat
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import prince
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as snsfrom 
from nicegui import ui
import json

def main():
    
    return

def convert_with_weights(df=None, column_name='milk'):
    OG_column_names = list(df.columns)
    df[column_name] = df[column_name].str.replace(" ", "")
    df[column_name] = df[column_name].str.replace("-", "")
    df = df.join(df.pop(column_name).str.get_dummies(','))
    column_names = list(df.columns)
    for item in OG_column_names:
        if item in column_names: column_names.remove(item)
    new_column_names = []
    for item in column_names:
        df = df.rename(columns = {item: f'{column_name}_{item}'})
        new_column_names.append(f'{column_name}_{item}')
    df['weights'] = 1/(df[new_column_names].sum(axis=1))
    df.replace([np.inf, -np.inf], 0, inplace=True)
    for item in new_column_names:
        df[item] = df[item] * df['weights']
    df = df.drop('weights', axis=1)
    return df

def convert_without_weights(df=None, column_name='column'):
    OG_column_names = list(df.columns)
    df[column_name] = df[column_name].str.replace(" ", "")
    df[column_name] = df[column_name].str.replace("-", "")
    df = df.join(df.pop(column_name).str.get_dummies(','))
    column_names = list(df.columns)
    for item in OG_column_names:
        if item in column_names: column_names.remove(item)
    for item in column_names:
        df = df.rename(columns = {item: f'{column_name}_{item}'})
    return df

def convert_string_to_number(equation):
    if '/' in equation:
        y = equation.split('/')
        x = float(y[0])/float(y[1])
    elif '-' in equation:
        x = (equation.replace("%", ""))
        y = x.split('-')
        x = ((float(y[0]) + float(y[1]))/2)
    elif '%' in equation:
        x = float(equation.replace("%", ""))/100
    return x

def convert_fat_content_to_percent(df, column_name = 'fat_content'):
    df[column_name] = df[column_name].str.replace(" ", "")
    df[column_name] = df[column_name].str.replace(r'[a-zA-Z]', '', regex=True)
    df[column_name] = df[column_name].map(lambda x: convert_string_to_number(x) if type(x) == str else x)
    return df

df = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2024/2024-06-04/cheeses.csv')
df = df.drop(['url', 'calcium_content', 'region', 'synonyms', 'alt_spellings'], axis=1)
cheese_df = df.copy()
file_path = 'country_climate_map.json' # Replace with the actual path to your JSON file
with open(file_path, 'r') as file:
    country_climate_map = json.load(file)

cheese_df = convert_with_weights(cheese_df, 'milk')
cheese_df["climate"] = cheese_df["country"].apply(lambda x: country_climate_map.get(x))
cheese_df = cheese_df.drop('country', axis=1)
#indicator variables? Look at top producers? Create lists that join them by latitiude and some that join them by longitude?
cheese_df = convert_without_weights(cheese_df, 'climate')
#Can these be seen as truth?  All the cheeses in the same family should be clustered together
# (cheese_df['cheese'].str.lower() == cheese_df['family'].str.lower()).sum()
# cheese_df = convert_without_weights(cheese_df, 'family')
#separate by comma and then create indicator variables
cheese_df = convert_without_weights(cheese_df, 'type')
#create function that converts the fractions into a percentage
cheese_df =convert_fat_content_to_percent(cheese_df)
cheese_df = convert_without_weights(cheese_df, 'texture')
cheese_df =convert_without_weights(cheese_df, 'rind')
cheese_df = convert_without_weights(cheese_df, 'color')
cheese_df = convert_with_weights(cheese_df, 'flavor')
cheese_df = convert_with_weights(cheese_df, 'aroma')
cheese_df['known_producer'] = 1 - cheese_df['producers'].isna().astype(int)
cheese_df = cheese_df.drop('producers', axis=1)
cheese_df[['vegan', 'vegetarian']] = cheese_df[['vegan', 'vegetarian']].fillna(False)
cheese_df['fat_content'] = cheese_df['fat_content'].fillna(0)
cheese_df[['fat_content']] = StandardScaler().fit_transform(cheese_df[['fat_content']])

# # Function to handle row submission
# def add_row():
#     new_row = {
#         'fat_content': fat_content_slider.value,
#         'vegetarian': vegetarian_checkbox.value,
#         'vegan': vegan_checkbox.value,
#         'known_producer': known_producer_checkbox.value,
#         'milk': milk_combo.value,
#         'climate': climate_combo.value,
#         'type': type_combo.value,
#         'texture': texture_combo.value,
#         'rind': rind_combo.value,
#         'color': color_combo.value,
#         'flavor': flavor_combo.value,
#         'aroma': aroma_combo.value
#     }
#     ui.notify(f'New row added: {pd.DataFrame(new_row)}')

# def find_cheese(new_row):
#     return

# with ui.column().classes('w-full items-center'):
#     # Float slider for fat_content
#     ui.label('Cheese Recommender').classes('text-xl')
#     ui.label('Fat Content')
#     fat_content_slider = ui.slider(min=0, max=1, step=0.01, value=0.5).props('label') \
#         .on('update:model-value').classes('w-1/2')

#     # Combo boxes with grouped options
#     milk_options = ['buffalo', 'camel', 'cow', 'donkey', 'goat', 'moose', 'plantbased', 'sheep', 'waterbuffalo', 'yak']
#     climate_options = ['Continental', 'Dry', 'Polar', 'Temperate', 'Tropical']
#     type_options = ['artisan', 'blueveined', 'brined', 'firm', 'freshfirm', 'freshsoft', 'hard', 'organic', 'processed', 'semifirm', 'semihard', 'semisoft', 'smearripened', 'soft', 'softripened', 'whey']
#     texture_options = ['brittle', 'buttery', 'chalky', 'chewy', 'close', 'compact', 'creamy', 'crumbly', 'crystalline', 'dense', 'dry', 'elastic', 'firm', 'flaky', 'fluffy', 'gooey', 'grainy', 'oily', 'open', 'runny', 'semifirm', 'smooth', 'soft', 'softripened', 'spreadable', 'springy', 'sticky', 'stringy', 'supple']
#     rind_options = ['artificial', 'ashcoated', 'bloomy', 'clothwrapped', 'edible', 'leafwrapped', 'moldripened', 'natural', 'plastic', 'rindless', 'washed', 'waxed']
#     color_options = ['blue', 'bluegrey', 'brown', 'brownishyellow', 'cream', 'goldenorange', 'goldenyellow', 'green', 'ivory', 'orange', 'palewhite', 'paleyellow', 'pinkandwhite', 'red', 'straw', 'white', 'yellow']
#     flavor_options = ['acidic', 'bitter', 'burntcaramel', 'butterscotch', 'buttery', 'caramel', 'citrusy', 'creamy', 'crunchy', 'earthy', 'floral', 'fruity', 'fullflavored', 'garlicky', 'grassy', 'herbaceous', 'lemony', 'licorice', 'meaty', 'mellow', 'mild', 'milky', 'mushroomy', 'nutty', 'oceanic', 'piquant', 'pronounced', 'pungent', 'rustic', 'salty', 'savory', 'sharp', 'smokey', 'smooth', 'sour', 'spicy', 'strong', 'subtle', 'sweet', 'tangy', 'tart', 'umami', 'vegetal', 'woody', 'yeasty']
#     aroma_options = ['aromatic', 'barnyardy', 'buttery', 'caramel', 'clean', 'earthy', 'fermented', 'floral', 'fresh', 'fruity', 'garlicky', 'goaty', 'grassy', 'herbal', 'lactic', 'lanoline', 'mild', 'milky', 'mushroom', 'musty', 'nutty', 'pecan', 'perfumed', 'pleasant', 'pronounced', 'pungent', 'rawnut', 'rich', 'ripe', 'smokey', 'spicy', 'stinky', 'strong', 'subtle', 'sweet', 'toasty', 'whiskey', 'woody', 'yeasty']

#     # Create combo boxes
#     milk_combo = ui.select(milk_options, label='Milk Type', multiple=True).classes('w-1/2')
#     climate_combo = ui.select(climate_options, label='Climate', multiple=False).classes('w-1/2')
#     type_combo = ui.select(type_options, label='Type', multiple=True).classes('w-1/2')
#     texture_combo = ui.select(texture_options, label='Texture', multiple=True).classes('w-1/2')
#     rind_combo = ui.select(rind_options, label='Rind', multiple=True).classes('w-1/2')
#     color_combo = ui.select(color_options, label='Color', multiple=False).classes('w-1/2')
#     flavor_combo = ui.select(flavor_options, label='Flavor', multiple=True).classes('w-1/2')
#     aroma_combo = ui.select(aroma_options, label='Aroma', multiple=True).classes('w-1/2')

#     # Checkboxes
#     with ui.row():
#         vegetarian_checkbox = ui.checkbox('Vegetarian')
#         vegan_checkbox = ui.checkbox('Vegan')
#         known_producer_checkbox = ui.checkbox('Known Producer')

#     # Submit button
#     ui.button('Find Cheese', on_click=add_row)

#     ui.run()

main()