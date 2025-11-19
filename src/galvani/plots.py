import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from disparity import estimate_disparity_px

b = 20
DATA_ROOT = "../../data/galvani"
dir = f'{DATA_ROOT}/output'
bilder = [
    f'{dir}/stereo_B{b}_left.png',
    f'{dir}/stereo_B{b}_right.png',
    f'{dir}/stereo_B{b}_side.png'
]

plt.figure(figsize=(18, 6), dpi=200)  # höhere Auflösung
for i, datei in enumerate(bilder):
    img = mpimg.imread(datei)
    plt.subplot(1, len(bilder), i + 1)
    plt.imshow(img)
    plt.axis('off')

plt.tight_layout()
plt.savefig(f"{dir}/plot_B{b}.png")

disp_stats = estimate_disparity_px(bilder[0], bilder[1])
print(disp_stats)