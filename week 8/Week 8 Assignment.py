
# For this assignment, I chose to follow the dataset presented in class and made several specification and more interactions for it. First, I changed the original all-blue columns for all continents into different color schemes. For Asia I changed to reds, Europe into green, Americas into pink or magenta, Oceania into light blue, and Africa into yellow, which each symbolizes orientalism, green party and environmentalism in Europe, drought and the Sahara in Africa, patriotism and passion in Americas, and Oceania with the Pacific,  (might be a little bit stereotypical). 

# I also realized that the table on the left is only sorted alphabetically without giving much details on the data itself (as it would always and only give countries like Afghanistan, Albania, Algeria...), so I also changed it with the most top 10s (of population, life expectancies, and GDP per capita) to see how it changes between countries and continents. 

# Moreover, the interactive graph only provides the total numbers for each continent without specification of the countries. In order to provide a detailed presentation of the data within each continent, I generated a second graph that provides the detailed data of the top 10s countries in each continent. By adding an interactive bar, the viewer/user can click on the bar and choose which continent data they want to look at. In addition, within the presentation of each continent, the color scheme not only follows the colorization based on the graph for continents above, but further adds the sequence from dark to light to indicate the descending order as well. The smaller issue is that for each continents color, it seems that the color scheme is not necesarrily consistent that Magenta only has 4 color sequences whereas red has 9. 

import wbdata
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np

from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# color mapping for continents
continent_colors = {
    'Asia': 'lightcoral',
    'Europe': 'lightgreen',
    'Americas': 'lightpink',
    'Oceania': 'lightblue',
    'Africa': 'lightyellow'
}

# Define color sequences for countries within each continent (for darker to lighter transition)
continent_color_sequences = {
    'Asia': px.colors.sequential.Reds[::-1],  
    'Europe': px.colors.sequential.Greens[::-1],
    'Americas': px.colors.sequential.Magenta[::-1],  
    'Oceania': px.colors.sequential.Blues[::-1],
    'Africa': px.colors.sequential.YlOrBr[::-1]
}

# Initialize the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(external_stylesheets=external_stylesheets)

# App layout
app.layout = html.Div([
    html.Div('World Bank Data Display',
             style={'textAlign': 'center', 'color': 'black', 'fontSize': 30}),

    html.Div([
        dcc.RadioItems(
            options=[{'label': i, 'value': i} for i in ['pop', 'lifeExp', 'gdpPercap']],
            value='lifeExp',
            inline=True,
            id='my-radio-buttons-final'
        )
    ]),

    html.Div([
        html.Div([
            dash_table.DataTable(id='table-container', page_size=11, style_table={'overflowX': 'auto'})
        ], className='six columns'),
        html.Div([
            dcc.Graph(id='histo-chart-final')
        ], className='six columns')
    ], className='row'),

    html.Div([
        dcc.Dropdown(
            id='continent-dropdown',
            options=[{'label': c, 'value': c} for c in df['continent'].unique()],
            value=df['continent'].unique()[0],
            clearable=False
        )
    ], style={'margin-top': '20px'}),

    html.Div([
        dcc.Graph(id='top10-chart')
    ])
])

# Add controls to build the interaction
@callback(
    [Output('histo-chart-final', 'figure'),
     Output('table-container', 'data'),
     Output('top10-chart', 'figure')],
    [Input('my-radio-buttons-final', 'value'),
     Input('continent-dropdown', 'value')]
)
def update_content(col_chosen, selected_continent):
    sorted_df = df.sort_values(by=col_chosen, ascending=False)

    # Main histogram graph (continent-level)
    fig = px.histogram(df, x='continent', y=col_chosen, histfunc='avg', 
                       color='continent', color_discrete_map=continent_colors)
    
    # Filter top 10 countries within selected continent
    top10_df = df[df['continent'] == selected_continent].nlargest(10, col_chosen)

    # Get a subset of colors (first 10 from reversed scale for dark-to-light effect)
    color_sequence = continent_color_sequences[selected_continent][:10]

    # Top 10 bar chart with dark-to-light coloring
    top10_fig = px.bar(
        top10_df, 
        x='country', 
        y=col_chosen, 
        title=f'Top 10 Countries in {selected_continent} by {col_chosen}', 
        color='country', 
        color_discrete_sequence=color_sequence,
    )
    top10_fig.update_layout(bargap=0.4)
    
    return fig, sorted_df.to_dict('records'), top10_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)




