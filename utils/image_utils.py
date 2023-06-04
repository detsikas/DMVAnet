import cv2
import matplotlib.pyplot as plt


def show_images(images, image_titles, grid=(1, 2), vmin=None, vmax=None, cmap='gray'):
    rows = grid[0]
    columns = grid[1]
    fig = plt.figure(figsize=(columns*6, rows*6))
    for index, (image, title) in enumerate(zip(images, image_titles)):
        axs = fig.add_subplot(rows, columns, index+1)
        axs.set_title(title)
        plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()


def scale_image(image, scale):
    scaled_image = cv2.resize(image.astype('uint8'), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    offset = (scaled_image.shape[0]-image.shape[0])//2
    cropped_image = scaled_image[offset:-offset, offset:-offset]
    return cropped_image[:256, :256]
