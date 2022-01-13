import model
import numpy

dataset = model.TuxDriverDataset("./dataset/training")

accumulator = numpy.array([0, 0, 0, 0])
empty_samples = 0

for data in dataset:
    sample = data[1].long().numpy()
    if not numpy.any(sample):
        empty_samples += 1
    else:
        accumulator = numpy.add(accumulator, sample)
    print("\r", accumulator, end="")

print("\r", accumulator)
print("Empty samples: ", empty_samples)
