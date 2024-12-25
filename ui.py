import tkinter as tk
from tkinter import Button
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance,ImageTk
import numpy as np
from model import Yolov3LoadPretrain, ScalePrediction, ResidualBlock, CNNBlock , Yolov3
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DrawingApp:
    def __init__(self, root, device):
        self.root = root
        self.root.title("Drawing App")

        self.canvas = tk.Canvas(root, width=640, height=640, bg='black')
        self.canvas.pack()

        self.image = Image.new('RGB', (640, 640), 'black')
        self.draw = ImageDraw.Draw(self.image)

        self.prev_x = None
        self.prev_y = None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.save_button = Button(root, text="Save Image", command=self.save_image)
        self.save_button.pack()
        self.device = device

    def paint(self, event):
        x, y = event.x, event.y
        if self.prev_x and self.prev_y:
            self.canvas.create_line(self.prev_x, self.prev_y, x, y, width=4, fill="white", capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.prev_x, self.prev_y, x, y], fill="white", width=4)
        self.prev_x = x
        self.prev_y = y

    def reset(self, event):
        self.prev_x = None
        self.prev_y = None
        
    def load_model(self,device):
        normalized_anchors = [[0.90479105, 0.89945908]
                            ,[0.74424541, 0.89958292]
                            ,[0.58524531, 0.86714259]
                            ,[1.3977123 , 1.39198349]
                            ,[1.03885399, 1.377646  ]
                            ,[0.6879216 , 1.6716429 ]
                            ,[2.0832272 , 2.13743838]
                            ,[1.40664571, 2.29924311]
                            ,[0.45904361, 0.52194387]]
        config = config = [
                        (512, 1, 1),
                        (1024, 3, 1),
                        "S",
                        (256, 1, 1),
                        "U",
                        (256, 1, 1),
                        (512, 3, 1),
                        "S",
                        (128, 1, 1),
                        "U",
                        (128, 1, 1),
                        (256, 3, 1),
                        "S",
                        ]
        yolo = Yolov3(device,normalized_anchors,config)
        yolo.load_state_dict(torch.load('yoloWeight.pth', map_location =self.device))
        self.model = yolo 
    
    def draw_bounding_boxes(self, image, score_threshold=0.85):
        # Get bounding boxes, scores, and class labels from the model
        boxes, scores, class_labels = self.model.tunningInput(image, score_threshold=score_threshold)

        # Clear the canvas
        self.canvas.delete("all")

        # Convert the image to a format compatible with tkinter
        resized_image = image.resize((640, 640), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Loop through the detected bounding boxes
        for box, label, score in zip(boxes.cpu(), class_labels.cpu(), scores.cpu()):
            x_min, y_min, x_max, y_max = box.tolist()

            # Scale bounding box coordinates to the canvas size
            x_min_scaled = x_min * 640 / image.width
            y_min_scaled = y_min * 640 / image.height
            x_max_scaled = x_max * 640 / image.width
            y_max_scaled = y_max * 640 / image.height

            # Draw the bounding box
            self.canvas.create_rectangle(
                x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled,
                outline="red", width=2
            )

            # Draw the label and score
            self.canvas.create_text(
                x_min_scaled, y_min_scaled - 10,
                anchor=tk.SW,
                text=f"Class {label.item()} ({score.item():.2f})",
                fill="red",
                font=("Helvetica", 10, "bold")
            )
      
    def save_image(self):
        enhanced_image = self.image.convert("L")
        
        enhancer = ImageEnhance.Brightness(enhanced_image)
        enhanced_image = enhancer.enhance(2.0)  # Tăng độ sáng
        
        threshold = 128
        binary_image = enhanced_image.point(lambda p: 255 if p > threshold else 0)

        output_image = binary_image.resize((160, 160), Image.Resampling.LANCZOS)

        # Save the image
        output_image.save("drawing.png", "PNG")
        
        # Draw bounding boxes on the canvas
        self.draw_bounding_boxes(image=output_image, score_threshold=0.85)
        print("Image saved as drawing.png")

if __name__ == "__main__":
    root = tk.Tk()
    device = torch.device('cpu')
    app = DrawingApp(root, device)
    app.load_model(device)
    root.mainloop()
