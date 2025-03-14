import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from bokeh.io import output_notebook, show
from bokeh.plotting import figure, from_networkx
from bokeh.models import (
    ColumnDataSource, LabelSet, MultiLine, 
    CategoricalColorMapper, Select, CustomJS
)
from bokeh.transform import factor_cmap
from bokeh.layouts import column
from bokeh.palettes import Category20, Category20c, Category10

# Load data
nodes = pd.read_csv("translated-node-revised.csv")
edges = pd.read_csv("translated_edge.csv")

# Filter only "person" category
nodes = nodes[nodes['category'] == 'person']

G = nx.DiGraph()

# Add filtered nodes
target_node_ids = set(nodes['id'])
for _, row in nodes.iterrows():
    G.add_node(row['id'], 
               label=row['label'],
               category=row['category'],
               info=row['info'],
               value=row['value'])

# Add edges only if both nodes are in the filtered set
filtered_edges = edges[(edges['source'].isin(target_node_ids)) & (edges['target'].isin(target_node_ids))]
for _, row in filtered_edges.iterrows():
    G.add_edge(row['source'], row['target'], label=row['label'])

# Compute node degrees
degrees = dict(nx.degree(G))
nx.set_node_attributes(G, name='degree', values=degrees)

# Community detection
communities = list(greedy_modularity_communities(G))
for node_id in G.nodes():
    for i, comm in enumerate(communities):
        if node_id in comm:
            G.nodes[node_id]['modularity'] = i

# Determine unique edge labels
unique_edge_labels = list(set(nx.get_edge_attributes(G, 'label').values()))
num_labels = len(unique_edge_labels)

# Assign colors to each unique label
palette = Category20c[20] if 20 in Category20c else Category20[20]  # Use 20-color Category20c if available

edge_color_mapper = CategoricalColorMapper(factors=unique_edge_labels, palette=palette)

# Define tooltips
TOOLTIPS = [
    ("Name", "@label"),
    ("Character Description", "@info{safe}"),
    ("Connection", "@degree"),
    ("Category", "@category")
]

# Create figure
plot = figure(title="Dream of the Red Chamber Network Analysis (Person Nodes)",
              tools="pan,wheel_zoom,box_zoom,save,reset,tap",
              tooltips=TOOLTIPS,
              active_scroll='wheel_zoom',
              width=1200, height=800)

layout_positions = nx.spring_layout(G, k=0.5, iterations=100, scale=1, center=(0,0))
graph = from_networkx(G, layout_positions)

# Edge color based on label
graph.edge_renderer.glyph = MultiLine(
    line_color={'field': 'label', 'transform': edge_color_mapper},
    line_alpha=0.6,
    line_width=1.5
)

# Generate labels for nodes
x = [layout_positions[node][0] for node in G.nodes()]
y = [layout_positions[node][1] for node in G.nodes()]

label_source = ColumnDataSource({
    'x': x,
    'y': y,
    'label': [data['label'] for _, data in G.nodes(data=True)]
})

labels = LabelSet(x='x', y='y', text='label',
                  source=label_source,
                  text_font_size='10px',
                  background_fill_color='white',
                  background_fill_alpha=0.7)
plot.add_layout(labels)

# Add edge labels (filtered for better visibility)
edge_labels = []
for source, target, data in G.edges(data=True):
    x_mid = (layout_positions[source][0] + layout_positions[target][0]) / 2
    y_mid = (layout_positions[source][1] + layout_positions[target][1]) / 2
    edge_labels.append((x_mid, y_mid, data['label']))

edge_label_source = ColumnDataSource({
    'x': [x for x, y, text in edge_labels],
    'y': [y for x, y, text in edge_labels],
    'label': [text for x, y, text in edge_labels]
})

edge_label_set = LabelSet(x='x', y='y', text='label',
                           source=edge_label_source,
                           text_font_size='8px',
                           background_fill_color='white',
                           background_fill_alpha=0.6)
plot.add_layout(edge_label_set)

plot.renderers.append(graph)

output_notebook()
show(plot)
