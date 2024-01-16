import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

class EdgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Edge Detection App")

        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=10, pady=10)

        self.load_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.kernel_button = tk.Button(self.root, text="Kernel", command=self.detect_edges_kernel)
        self.kernel_button.pack(pady=5)

        self.sobel_button = tk.Button(self.root, text="Sobel", command=self.detect_edges_sobel)
        self.sobel_button.pack(pady=5)

        self.laplacian_button = tk.Button(self.root, text="Laplacian", command=self.detect_edges_laplacian)
        self.laplacian_button.pack(pady=5)

        self.canny_button = tk.Button(self.root, text="Canny", command=self.detect_edges_canny)
        self.canny_button.pack(pady=5)

        self.enbossing_button = tk.Button(self.root, text="Enbossing", command=self.detect_edges_enbossing)
        self.enbossing_button.pack(pady=5)

        self.blur_button = tk.Button(self.root, text="Motion Blur", command=self.apply_motion_blur)
        self.blur_button.pack(pady=5)

        self.image_path = ""
        self.image = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(file_path)
            self.display_image()

    def display_image(self):
        if self.image is not None:
            # Resize image to a smaller size for display
            resized_image = cv2.resize(self.image, (300, 300))

            image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)
            self.image_label.config(image=image_tk)
            self.image_label.image = image_tk

    def detect_edges_kernel(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Kernel
            kernel = cv2.getDerivKernels(1, 0, 3, normalize=True)
            kernel_image = cv2.filter2D(gray_image, -1, kernel[0] * kernel[1].T)

            # Display the result
            self.display_image_with_popup(kernel_image, "Kernel")

    def detect_edges_sobel(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Sobel
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_image = cv2.magnitude(sobel_x, sobel_y)

            # Display the result
            self.display_image_with_popup(sobel_image, "Sobel")

    def detect_edges_laplacian(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Laplacian
            laplacian_image = cv2.Laplacian(gray_image, cv2.CV_64F)

            # Display the result
            self.display_image_with_popup(laplacian_image, "Laplacian")

    def detect_edges_canny(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Blur the image to reduce noise
            gray_image_blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

            # Canny edge detection with adjusted thresholds
            canny_image = cv2.Canny(gray_image_blurred, 30, 90, apertureSize=3)

            # Display the result
            self.display_image_with_popup(canny_image, "Canny")

    def detect_edges_enbossing(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Enbossing
            kernel_enbossing = np.array([[0, -1, -1],
                                         [1, 0, -1],
                                         [1, 1, 0]])
            enbossing_image = cv2.filter2D(gray_image, -1, kernel_enbossing)

            # Display the result
            self.display_image_with_popup(enbossing_image, "Enbossing")

    def apply_motion_blur(self):
        if self.image is not None:
            # Create a motion blur kernel
            kernel_size = 15
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel_motion_blur /= kernel_size

            # Apply the motion blur kernel
            motion_blur_image = cv2.filter2D(self.image, -1, kernel_motion_blur)

            # Display the result
            self.display_image_with_popup(motion_blur_image, "Motion Blur")

    def display_image_with_popup(self, image, method_name):
        # Chuyển đổi kiểu dữ liệu về uint8 nếu ảnh có kiểu dữ liệu là float64
        if image.dtype == np.float64:
            image = cv2.convertScaleAbs(image)

        # Chuyển đổi ảnh sang định dạng RGB nếu nó là ảnh mức xám
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Hiển thị ảnh sử dụng matplotlib
        plt.imshow(image)
        plt.title(method_name)
        plt.show()




if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeDetectionApp(root)
    root.mainloop()
