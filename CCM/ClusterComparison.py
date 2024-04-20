from sklearn import cluster
from Infrastructure import GenerateDataAndDisplayPlots

# TODO: Define at least 6 different agglomerative models
example_model_name = "Example Model"
example_model = cluster.AgglomerativeClustering()

# Define a list of all your models as tuples (model name/label, model instance)
clustering_algorithms = [
    (example_model_name, example_model),
    ("Ward Linkage", cluster.AgglomerativeClustering(n_clusters=3, linkage='ward')),
    ("Complete Linkage", cluster.AgglomerativeClustering(n_clusters=3, linkage='complete')),
    ("Average Linkage", cluster.AgglomerativeClustering(n_clusters=3, linkage='average')),
    ("Single Linkage", cluster.AgglomerativeClustering(n_clusters=3, linkage='single')),
    ("Manhattan Linkage", cluster.AgglomerativeClustering(n_clusters=3, linkage='average', affinity='manhattan')),
    ("Cosine Linkage", cluster.AgglomerativeClustering(n_clusters=3, linkage='average', affinity='cosine'))
    # TODO: Add More Models
]

# Run the data-generation and plotting functions
GenerateDataAndDisplayPlots(clustering_algorithms)
