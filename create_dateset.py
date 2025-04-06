import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

# Step 1: Load a dataset from FiftyOne Zoo
dataset = foz.load_zoo_dataset("quickstart")

# Step 2: Compute embeddings and create a MongoDB similarity index
mongodb_index = fob.compute_similarity(
    dataset,
    embeddings="embeddings",  # the field in which to store the embeddings
    brain_key="mongodb_index",
    backend="mongodb",
)

# Wait for the index to be ready for querying
assert mongodb_index.ready

# Step 3: Query the dataset by similarity
query = dataset.first().id  # Query by sample ID
view = dataset.sort_by_similarity(
    query,
    brain_key="mongodb_index",
    k=10,  # Limit to 10 most similar samples
)

# Print the IDs of the most similar samples
print("Most similar samples:")
for sample in view:
    print(sample.id)

# Step 4: Cleanup
mongodb_index.cleanup()  # Delete the MongoDB vector search index
dataset.delete_brain_run("mongodb_index")  # Delete the brain run record
