import pandas as pd
import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from keras import layers, Model, optimizers, losses, metrics
from sklearn.model_selection import train_test_split
import numpy as np
import random

'''
# Uncomment once csv files are made
# Create edges data
# define columns
columns = ["edge_id", "source", "target", "weight"] + [f"H{i}" for i in range(1, 25)]
df = pd.DataFrame(columns=columns)

edge_counter = 0  # Initialize edge counter

for i in range(1, 21):  # 20 edges
    for j in range(24):  # Edge for each hour
        edge_id = edge_counter  # Assigning unique edge_id
        source = i
        target = random.randint(1, 20)
        weight = round(random.uniform(0, 1), 2)  # Weight between 0 and 1
        hour = [0] * 24
        hour[j] = 1  # Set the hour corresponding to the edge to 1
        df.loc[edge_counter * 24 + j] = [edge_id, source, target, weight] + hour
        edge_counter += 1  # Increment edge counter

df.to_csv("C:/Users/ibrah/PycharmProjects/UmrahAI-AIprototype/edges.csv", sep=",", index=False)
print(df)


# Create nodes data
edges_df = pd.read_csv("C:/Users/ibrah/PycharmProjects/UmrahAI-AIprototype/edges.csv")
neighbors_count = edges_df['source'].value_counts() + edges_df['target'].value_counts()
nodes_df = pd.DataFrame({"node_id": neighbors_count.index, "num_neighbors": neighbors_count.values})
nodes_df.to_csv("C:/Users/ibrah/PycharmProjects/UmrahAI-AIprototype/node.csv", index=False)
'''

edge_dataset_path = "C:/Users/ibrah/PycharmProjects/UmrahAI-AIprototype/edges.csv"  # replace these with absolute path
node_dataset_path = "C:/Users/ibrah/PycharmProjects/UmrahAI-AIprototype/node.csv"  # of csv file in your system


edge_dataset = pd.read_csv(
    edge_dataset_patw
    sep=",",  # comma-separated
    header=None,  # no heading row
    names=["edge_id", "source", "target", "weight", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10",
           "H11", "H12", "H13", "H14", "H15", "H16", "H17", "H18", "H19", "H20", "H21", "H22", "H23", "H24"],  # set our own names for the columns
)
edge_dataset = edge_dataset.set_index("edge_id")
print(edge_dataset, "\n\n")


node_dataset = pd.read_csv(
    node_dataset_path,
    sep=",",  # comma-separated
    header=None,  # no heading row
    names=["node_id", "num_neighbors"],
)
node_dataset = node_dataset.set_index("node_id")
print(node_dataset, "\n\n")



stellargraph = sg.StellarGraph(edges=edge_dataset, nodes=node_dataset)
print(stellargraph.info())


import networkx as nx

def create_graph_from_df(edge_dataset):
    G = nx.DiGraph()
    for index, row in edge_dataset.iterrows():
        source = row['source']
        target = row['target']
        for hour in range(1, 25):
            if row['H' + str(hour)] == 1:  # If the edge is active at this hour
                weight = row['weight']  # The crowd level is the weight of the edge
                # Add the edge to the graph with the hour and crowd level as the weight
                G.add_edge(source, target, hour=hour, weight=weight)
    return G

G = create_graph_from_df(edge_dataset)
print(G.nodes())
def find_best_path_and_time(G, start_node, end_node):
    # Initialize the best path and minimum crowd level
    best_path = None
    min_crowd_level = float('inf')
    best_hour = None

    # Iterate over all hours
    for hour in range(1, 25):
        # Create a subgraph for the current hour
        H = G.edge_subgraph((u, v) for u, v, d in G.edges(data=True) if d['hour'] == hour)

        # Check if a path exists
        if nx.has_path(H, start_node, end_node):
            # Find the shortest path in the subgraph
            path = nx.dijkstra_path(H, start_node, end_node, weight='weight')

            # Calculate the total crowd level for the path
            total_crowd_level = sum(H[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))

            # If the total crowd level is less than the current minimum, update the best path and minimum crowd level
            if total_crowd_level < min_crowd_level:
                best_path = path
                min_crowd_level = total_crowd_level
                best_hour = hour

    return best_path, min_crowd_level, best_hour

# Test the function
start_node = 7.0
end_node = 6.0
best_path, min_crowd_level, best_hour = find_best_path_and_time(G, start_node, end_node)
print(f"The best path from node {start_node} to node {end_node} at hour {best_hour} is {best_path} with a total crowd level of {min_crowd_level}.")


'''
# Split the edge data into train and test sets
edge_ids_train, edge_ids_test = train_test_split(edge_dataset[["source", "target"]].values, train_size=0.8, test_size=0.2)

# Create a generator object for training and testing
batch_size = 32
generator = sg.mapper.GraphSAGELinkGenerator(stellargraph, batch_size=batch_size, num_samples=[10, 5])

# Build the GraphSAGE model
graphsage = GraphSAGE(
    layer_sizes=[32, 32],
    generator=generator,
    bias=True,
    dropout=0.5,
)

# Define the model architecture
x_inp, x_out = graphsage.in_out_tensors()
x_out_concat = layers.Concatenate(axis=1)(x_out)  # Concatenate the output tensors
x_out_dense = layers.Dense(units=32, activation="relu")(x_out_concat)  # Add a Dense layer
prediction = layers.Dense(units=1, activation="relu")(x_out_dense)  # Final prediction layer


# Create the Keras model
model = Model(inputs=x_inp, outputs=prediction)

# Compile the model
model.compile(
    optimizer=optimizers.Adam(lr=0.01),
    loss=losses.mean_squared_error,
    metrics=[metrics.mean_absolute_error]
)

# Train the model
history = model.fit(
    generator.flow(edge_ids_train),  # Use generator to get the training data
    epochs=20,
    validation_data=generator.flow(edge_ids_test),  # Use generator to get the validation data
    verbose=1
)


# Evaluate the model
test_metrics = model.evaluate(generator.flow(edge_ids_test))
print("\nTest Mean Squared Error:", test_metrics[0])
print("\nTest Mean Absolute Error:", test_metrics[1])


#
# Split the edge data into train and test sets
edge_ids_train, edge_ids_test, edge_labels_train, edge_labels_test = train_test_split(
    edge_dataset[["source", "target"]].values,
    edge_dataset["weight"].values,
    train_size=0.8,
    test_size=0.2,
    stratify=edge_dataset["weight"]  # Stratify based on edge weights to maintain their distribution
)

# Create a generator object for training and testing
batch_size = 32
generator = sg.mapper.GraphSAGELinkGenerator(stellargraph, batch_size, [10, 5])

# Build, compile, fit, and evaluate the model as before
graphsage = GraphSAGE(
    layer_sizes=[32, 32],
    generator=generator,
    bias=True,
    dropout=0.5,
)

x_inp, x_out = generator.flow(edge_ids_test)

model = Model(inputs=x_inp, outputs=x_out)
model.compile(
    optimizer=optimizers.Adam(lr=0.01),
    loss=losses.mean_squared_error,
    metrics=[metrics.mean_absolute_error],
)

history = model.fit(
    generator.flow(edge_ids_train, edge_labels_train),
    epochs=20,
    validation_data=generator.flow(edge_ids_test, edge_labels_test),
    verbose=1,
)

test_metrics = model.evaluate(
    generator.flow(edge_ids_test, edge_labels_test)
)
print("\nTest Mean Squared Error:", test_metrics[0])
print("\nTest Mean Absolute Error:", test_metrics[1])


# Use the function to find the best path as before
best_path, best_hour, total_weight = find_best_path(1, 10, model)
print(f"The best path from node {start_node} to node {end_node} is {best_path}")
print(f"The best hour to take this path is {best_hour} o'clock")
print(f"The total weight of this path is {total_weight:.2f}")
'''

