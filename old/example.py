from old.dataset import *
from old.beta import *

# Let us visualize the output image at a few timestamps
sample_coco = next(iter(dataset))[0]

fig = plt.figure(figsize=(15, 30))

for index, i in enumerate([10, 100, 150, 199]):
    noisy_im, noise = forward_noise(0, np.expand_dims(sample_coco, 0), np.array([i, ]))
    plt.subplot(1, 4, index + 1)
    plt.imshow(np.squeeze(np.squeeze(noisy_im, -1), 0), cmap='gray')

plt.show()
