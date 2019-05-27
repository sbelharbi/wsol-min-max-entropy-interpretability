# Main references:
# https://github.com/Peter554/StainTools

import datetime as dt
from os.path import join
import timeit

from stain_tools.visualization import read_image, make_image_stack, plot_image_stack
from stain_tools.tools import BrightnessStandardizer
from stain_tools.stain_normalizer import StainNormalizer
from stain_tools.stain_augmentor import StainAugmentor

# Set up
METHOD = "macenko"  # "macenko", "vahadane"
STANDARDIZE_BRIGHTNESS = True
RESULTS_DIR = join("./results/", str(dt.datetime.now()))
REPEAT = 1


# Read the images.
i1 = read_image("./data/png/i1.png")
i2 = read_image("./data/png/i2.png")
i3 = read_image("./data/png/i3.png")
i4 = read_image("./data/png/i4.png")
i5 = read_image("./data/png/i5.png")

# ======
# Timing
# ======


def wrapper1():
    return read_image("./data/png/i1.png")


time = timeit.timeit(wrapper1, number=REPEAT) / REPEAT
print("Avg.time: read_image() --> ", time)



# Plot
stack = make_image_stack([i1, i2, i3, i4, i5])
titles = ["Target"] + ["Original"] * 4
plot_image_stack(stack, width=5, title_list=titles, save_name=RESULTS_DIR + "original-images.png", show=0)

# ========================================================
# Brightness standardization
# (Can be skipped but can help with tissue mask detection)
# ========================================================

if STANDARDIZE_BRIGHTNESS:
    standardizer = BrightnessStandardizer()
    i1 = standardizer.transform(i1)
    i2 = standardizer.transform(i2)
    i3 = standardizer.transform(i3)
    i4 = standardizer.transform(i4)
    i5 = standardizer.transform(i5)

    # Plot
    stack = make_image_stack([i1, i2, i3, i4, i5])
    titles = ["Target standardized"] + ["Original standardized"] * 4
    plot_image_stack(
        stack, width=5, title_list=titles, save_name=RESULTS_DIR + "original-images-standardized.png", show=0)

    # ======
    # Timing
    # ======
    def wrapper2():
        return standardizer.transform(i1)


    time = timeit.timeit(wrapper2, number=REPEAT) / REPEAT
    print("Avg.time: standardizer.transform() --> ", time)

# ===================
# Stain normalization
# ===================

# Normalize to the stain of the first image.
normalizer = StainNormalizer(method=METHOD)
normalizer.fit(i1)
i2_normalized = normalizer.transform(i2)
i3_normalized = normalizer.transform(i3)
i4_normalized = normalizer.transform(i4)
i5_normalized = normalizer.transform(i5)

# Plot
stack = make_image_stack([i1, i2_normalized, i3_normalized, i4_normalized, i5_normalized])
titles = ["Target"] + ["Stain normalized"] * 4
plot_image_stack(
    stack, width=5, title_list=titles, save_name=RESULTS_DIR + "-stain-normalized-images-" + METHOD + ".png", show=0)


# ======
# Timing
# ======


def wrapper3():
    return normalizer.fit(i1)


time = timeit.timeit(wrapper3, number=REPEAT) / REPEAT
print("Avg.time: normalizer.fit() " + METHOD + " --> ", time)


# ======
# Timing
# ======


def wrapper4():
    return normalizer.transform(i5)


time = timeit.timeit(wrapper4, number=REPEAT) / REPEAT
print("Avg.time: normalizer.transform() " + METHOD + " --> ", time)

# ==================
# Stain augmentation
# ==================

# Augment the first image using pop().
augmentor = StainAugmentor(method=METHOD, sigma1=0.4, sigma2=0.4)
augmentor.fit(i1)

augmened_images = []
for _ in range(10):
    augmened_image = augmentor.pop()
    augmened_images.append(augmened_image)

# Plot
stack = make_image_stack([i1] + augmened_images)
titles = ["Original"] + ["Augmented-pop()"] * 10
plot_image_stack(
    stack, width=5, title_list=titles, save_name=RESULTS_DIR + 'stain-augmented-images-' + METHOD + ".png", show=0)


# Augment the first image using augment().
augmentor2 = StainAugmentor(method=METHOD, sigma1=0.4, sigma2=0.4)

augmened_images = []
for _ in range(10):
    augmened_image = augmentor2.augment(i1)
    augmened_images.append(augmened_image)

# Plot
stack = make_image_stack([i1] + augmened_images)
titles = ["Original"] + ["Augmented-augment()"] * 10
plot_image_stack(
    stack, width=5, title_list=titles, save_name=RESULTS_DIR + 'stain-augmented-images-' + METHOD + ".png", show=0)

# ======
# Timing
# ======


def wrapper5():
    return augmentor.pop()


time = timeit.timeit(wrapper5, number=REPEAT) / REPEAT
print("Avg.time: augmentor.pop() --> ", time)
