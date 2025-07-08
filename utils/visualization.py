from matplotlib import pyplot as plt

def show_image(img, title="Image"):
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()
