import torch
import crypten
import crypten.mpc
from PIL import Image
import numpy as np
import os

# Initialize CrypTen
crypten.init()


class SecureAverageFilter:
    def __init__(self, num_parties=3):
        self.num_parties = num_parties

    def simulate_mpc(self, image_tensors):
        """
        Simulate MPC computation for an average filter on three images
        Args:
            image_tensors (list of torch.Tensor): List of image tensors from different parties
        Returns:
            torch.Tensor: Smoothed image tensor (average of input images)
        """
        # Ensure all input tensors are encrypted
        encrypted_images = [crypten.cryptensor(img) for img in image_tensors]

        # Perform secure computation: pixel-wise sum and average
        encrypted_sum = sum(encrypted_images)
        encrypted_avg = encrypted_sum / self.num_parties

        # Decrypt the final smoothed image
        smoothed_image = encrypted_avg.get_plain_text()

        return smoothed_image

#Helper function for loading images
def load_image_as_tensor(image_path):
    """
    Loads an image and converts it to a PyTorch tensor
    Args:
        image_path (str): Path to the image file
    Returns:
        torch.Tensor: Image tensor normalized to [0, 1]
    """
    # Open image and convert to grayscale for simplicity
    image = Image.open(image_path).convert('L')
    
    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image) / 255.0

    # Convert to PyTorch tensor
    return torch.tensor(image_array, dtype=torch.float)

#Demo
def run_average_filter_demo():
    # Example image paths (in practice, these would be private inputs from parties)
    image_paths = [
        "images_bundle/cameraman.tif",  # Party 1's image
        "images_bundle/cameraman.tif",  # Party 2's image
        "images_bundle/cameraman.tif"   # Party 3's image
    ]

    print("Loading images from parties...")
    # Load and display the original images
    images = [load_image_as_tensor(path) for path in image_paths]
    for i, img in enumerate(images, 1):
        print(f"Image from Party {i} loaded with shape: {img.shape}")

    print("\nStarting secure average filter computation...")

    # Create average filter instance
    avg_filter = SecureAverageFilter()

    # Perform secure computation
    smoothed_image = avg_filter.simulate_mpc(images)

    print("\nSecure computation completed. Displaying results...")

    # Convert the smoothed image back to a displayable format
    smoothed_image = (smoothed_image * 255).numpy().astype(np.uint8)
    smoothed_image = Image.fromarray(smoothed_image)
    #smoothed_image.show(title="Smoothed Image")
    return smoothed_image


# Save the smoothed image as a PNG file
def save_smoothed_image(smoothed_image, output_path):
    """
    Save the smoothed image tensor as a PNG file.
    Args:
        smoothed_image_tensor (torch.Tensor): The smoothed image tensor.
        output_path (str): Path where the image will be saved.
    """
    # Convert tensor to a PIL image (ensure it's in the correct format)
    #smoothed_image_array = smoothed_image_tensor.numpy()  # Convert to numpy array
    #smoothed_image_array = (smoothed_image_array * 255).astype('uint8')  # Scale to [0, 255]
    #smoothed_image = Image.fromarray(smoothed_image_array)

    # Save the image
    smoothed_image.save(output_path)
    print(f"Smoothed image saved at: {output_path}")


#Run demo
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
 
    smoothed_image = run_average_filter_demo()

    output_path = "output_images/smoothed_image.png"

    # Assuming `smoothed_image` is the resulting tensor from the average filter
    save_smoothed_image(smoothed_image, output_path)


if __name__ == "__main__":
    main()
