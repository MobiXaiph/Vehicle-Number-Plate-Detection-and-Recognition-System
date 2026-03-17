import matplotlib.pyplot as plt
import cv2

def plot_images(images, titles, cmap='gray', figsize=(15, 5)):
    """
    Plots a list of images with titles.
    
    Args:
        images: List of images (numpy arrays).
        titles: List of titles for each image.
        cmap: Colormap to use (default 'gray').
        figsize: Figure size.
    """
    count = len(images)
    plt.figure(figsize=figsize)
    for i in range(count):
        plt.subplot(1, count, i + 1)
        if len(images[i].shape) == 3:
            # Convert BGR to RGB for display if it's a color image
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('debug_output.png')
    # plt.show()

def show_result(image, text, confidence):
    """
    Displays the final result with the detected text.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected: '{text}' ({confidence:.2f})")
    plt.axis('off')
    plt.savefig('result_output.png')
    # plt.show()
