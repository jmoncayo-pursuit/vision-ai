import fiftyone as fo

dataset = fo.Dataset(name="test")

for i in range(10):

    sample = fo.Sample(
        filepath="test.jpg",

    )
    dataset.add_sample(sample)


session = fo.launch_app(dataset)

for sample in dataset:

    sample["ground_truth"] = fo.Detections(
        detections=[
            fo.Detection(label="cat", bounding_box=[0.1, 0.1, 0.5, 0.5]),
            fo.Detection(label="dog", bounding_box=[0.6, 0.6, 0.9, 0.9]),
        ]
    )
    sample.save()
