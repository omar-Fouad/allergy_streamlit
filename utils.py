import random
from PIL import Image
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import os
import numpy as np
import pyrealsense2 as rs
import cv2
import socket
import pickle
#from tkinter import Tk, filedialog


def start_stream(name=None):
    pipeline = rs.pipeline()
    config = rs.config()
   # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
   # pipeline.start(config)
   # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    if name:
     rs.config.enable_device_from_file(config,name)

    
    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)

    # Start streaming from file
    pipeline.start(config)

    return pipeline

# Function to stop the RealSense pipeline
def stop_stream(pipeline):
    pipeline.stop()

# Function to get a frame from the RealSense pipeline
def get_frame(pipeline):
    # frames = pipeline.wait_for_frames()
    # color_frame = frames.get_color_frame()
    # if not color_frame:
        # return None
    # color_image = np.asanyarray(color_frame.get_data())
        # Create colorizer object
        colorizer = rs.colorizer()

        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())


        return depth_color_image


def paginate_images(images, page, per_page):
    """Paginate the list of images."""
    start = (page - 1) * per_page
    end = start + per_page
    return images[start:end]

def save_uploaded_file(uploaded_dir, uploaded_file):
    file_path = os.path.join(uploaded_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def save_captured_image(image, uploaded_dir, file_name):
    file_path = os.path.join(uploaded_dir, file_name)
    image.save(file_path)
    return file_path

def load_images(uploaded_dir):
    """Load all images from the upload directory."""
    return [f for f in os.listdir(uploaded_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

def delete_image(uploaded_dir, image_name):
    """Delete an image by name."""
    file_path = os.path.join(uploaded_dir, image_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        
def delete_images(uploaded_dir, selected_images):
    """Delete multiple images by names."""
    for image_name in selected_images:
        file_path = os.path.join(uploaded_dir, image_name)
        if os.path.exists(file_path):
            os.remove(file_path)
        

def capture_from_realsense(st):
    """Capture an image from the Intel RealSense camera."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            st.error("No frame captured from the camera.")
            return None

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        image = Image.fromarray(color_image)
        return image
    finally:
        pipeline.stop()   

def     capture_from_webcam(st):
    """Capture an image from the laptop's webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the webcam.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Failed to capture image from webcam.")
        return None

    # Convert BGR (OpenCV format) to RGB (PIL format)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return image

def create_allergen_overlay(image_path, allergens):
    image = Image.open(image_path)

    # Resize image to reduce size (optional)
    image = image.resize((300, 400))  # Adjust width and height as needed
    draw = ImageDraw.Draw(image)
    
    # Font setup
    try:
        font = ImageFont.truetype("arial.ttf", size=12)  # Smaller font size for readability
    except IOError:
        font = ImageFont.load_default()

    # Adjusted coordinates for pins
    left_pins = [(10, y) for y in range(50, 400, 70)]  # Increased vertical space
    right_pins = [(160, y) for y in range(50, 400, 70)]  # Adjusted for the resized image

    # Divide allergens into left and right groups
    left_allergens = allergens[:5]  # First 5 allergens
    right_allergens = allergens[5:10]  # Next 5 allergens


    # Overlay text on the left pins
    for idx, coord in enumerate(left_pins):
        if idx < len(left_allergens):  # Ensure index is within range
            text = left_allergens[idx]
            text_bbox = draw.textbbox((0, 0), text, font=font)  # Get bounding box
            text_width = text_bbox[2] - text_bbox[0]  # Calculate text width
            adjusted_coord = (coord[0] + 140 - text_width - 10, coord[1])  # Middle - text_width - 10
            draw.text(adjusted_coord, text, fill="blue", font=font)

    for idx, coord in enumerate(right_pins):
        if idx < len(right_allergens):  # Ensure index is within range
            draw.text(coord, right_allergens[idx], fill="red", font=font)

    return image

# Function to generate PDF
def generate_pdf(title, product_name, patient_info, physician_info, results):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title and Product Name
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(width / 2, height - 40, title)
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(width / 2, height - 60, f"Product: {product_name}")

    # Constants for layout
    section_height = 60  # Adjusted height for patient/physician sections
    table_spacing = 20   # Space between tables and sections
    
    # Practitioner Signature and Date
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, height - 90, "Practitioner Signature: ____________________")
    pdf.drawString(400, height - 90, "Date: ____________________")

    # Patient Section with Grid and Continuous Line
    pdf.setStrokeColor(colors.black)
    pdf.setFillColor(colors.lightblue)
    pdf.rect(45, height - section_height - 120, width - 90, section_height, stroke=1, fill=1)
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, height - 130, "Patient Information:")

    # Drawing continuous lines (slightly lower)
    line_y = height - 140  # Adjust the Y-coordinate for lines
    pdf.line(50, line_y, width - 45, line_y)  # Continuous line for the section
    pdf.line(50, line_y - 20, width - 45, line_y - 20)  # Additional line for spacing

    # Patient Info Grid (Left Column)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(60, line_y - 15, f"Name: {patient_info['name']}")
    pdf.drawString(60, line_y - 35, f"ID: {patient_info['id']}")

    # Patient Info Grid (Right Column)
    pdf.drawString(width / 2, line_y - 15, f"Date of Birth: {patient_info['dob']}")
    pdf.drawString(width / 2, line_y - 35, f"Location: {patient_info['location']}")

    # Physician Section with Grid and Continuous Line
    pdf.setStrokeColor(colors.black)
    pdf.setFillColor(colors.lightblue)
    pdf.rect(45, height - section_height - 210, width - 90, section_height, stroke=1, fill=1)
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, height - 220, "Physician Information:")

    # Drawing continuous lines (slightly lower)
    line_y_physician = height - 230  # Adjust the Y-coordinate for lines
    pdf.line(50, line_y_physician, width - 45, line_y_physician)  # Continuous line for the section
    pdf.line(50, line_y_physician - 20, width - 45, line_y_physician - 20)  # Additional line for spacing

    # Physician Info Grid (Left Column)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(60, line_y_physician - 15, f"Name: {physician_info['name']}")
    pdf.drawString(60, line_y_physician - 35, f"Address: {physician_info['address']}")

    # Physician Info Grid (Right Column)
    pdf.drawString(width / 2, line_y_physician - 15, f"Telephone: {physician_info['telephone']}")
    pdf.drawString(width / 2, line_y_physician - 35, f"Email: {physician_info['email']}")


    # Tables
    panel_titles = ["Panel A", "Panel B", "Panel C", "Panel D"]
    current_y = height - section_height - 300  # Position tables below the blue sections

    for i, panel in enumerate(panel_titles):
        # Adjust table position for two columns
        x_offset = 45 if i % 2 == 0 else width / 2 + 20

        # Move to the next row if two tables are in the same line
        if i % 2 == 0 and i != 0:
            current_y -= 200 + table_spacing  # Adjust spacing between rows

        # Table Data: Including Panel Title as the First Row
        data = [[panel, "", ""]]  # Panel title spans all columns
        data.append(["Index", "Allergen", "Well (mm)"])  # Column headers

            
        # Only add data if the panel has entries
        if panel in results and results[panel]:
            for site in results[panel]:
                data.append([site["index"], site["allergen"], site["well"]])
        else:
            data.append(["-", "-", "-"])  # Placeholder for empty panels

        # Table Styling
        table = Table(data, colWidths=[50, 140, 50])
        table.setStyle(TableStyle([
            ('SPAN', (0, 0), (-1, 0)),  # Span panel title across all columns
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),  # Panel title background
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Panel title text color
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),  # Center-align panel title
            ('BACKGROUND', (0, 1), (-1, 1), colors.lightblue),  # Header background
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.black),  # Header text color
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BACKGROUND', (0, 2), (-1, -1), colors.whitesmoke)
        ]))
        table.wrapOn(pdf, 400, 200)
        table.drawOn(pdf, x_offset, current_y - 150)  # Adjust the Y position for the table


    pdf.save()
    buffer.seek(0)
    return buffer

def load_model(model_name: str, encoder_name: str, checkpoint_path: str, device: torch.device):
    """Load the segmentation model from a checkpoint."""
    model = getattr(smp, model_name)(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1).to(
        device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state'])
    model.eval()
    return model


def segment_image(model, image_path, transform, device):
    """Apply segmentation on a single image."""
    from PIL import Image  # Ensure PIL is imported
    image = Image.open(image_path).convert("RGB")  # Open and convert image to RGB
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).cpu().squeeze().numpy() > 0.5
    return image, pred_mask



def save_segmented_image(original_image, segmented_mask, output_path):
    """Save the original image alongside the segmented mask."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_mask, cmap="gray")
    plt.title("Segmented Mask")
    plt.axis("off")
    plt.savefig(output_path)  # Save to the specified output path
    plt.close()



def play_sound_html(sound_path, st):
    """
    Embeds an HTML5 audio player to play sound automatically.
    """
    sound_html = f"""
    <audio autoplay>
        <source src="data:audio/mpeg;base64,{sound_path}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(sound_html, unsafe_allow_html=True)

# Convert sound file to base64
def get_base64_sound(file_path):
    """
    Reads a sound file and converts it to a base64 string.
    """
    import base64
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode()


# Mock AI function (you can expand this with actual image processing later)
def mock_ai_verification():
    # Placeholder logic for image verification
    # Ideally, you could integrate real AI/ML-based verification here.
    return "All items are verified and present in the kit!"

# Mock function to simulate template matching
def mock_template_matching():
    # Simulate analysis results
    matched = random.choice([True, False])
    details = {
        "status": "Correctly Placed" if matched else "Misaligned",
        "suggestion": "No adjustment needed." if matched else "Reposition stickers for accuracy."
    }
    return details
    

def resize_image(image_path, width, height=None):

    with Image.open(image_path) as img:
        if height is None:
            # Calculate height while maintaining the aspect ratio
            aspect_ratio = img.height / img.width
            height = int(width * aspect_ratio)
        resized_img = img.resize((width, height))
    return resized_img

class RealSenseClient:
    def __init__(self, host='https://1837-109-236-81-182.ngrok-free.app', port=5000):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))
        print(f"Connected to server at {self.host}:{self.port}")
    def destroy(self):
        self.client_socket.close()

    def receive_data(self):
        try:
            while True:
                # Receive the size of the incoming data
                size_data = self.client_socket.recv(4)
                if not size_data:
                    break
                size = int.from_bytes(size_data, byteorder='big')

                # Receive the serialized data
                serialized_data = b''
                while len(serialized_data) < size:
                    serialized_data += self.client_socket.recv(size - len(serialized_data))

                # Deserialize the data
                data = pickle.loads(serialized_data)
                depth_image = data['depth']
                color_image = data['color']

                # Display the received frames
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                return color_image, depth_image
                # cv2.imshow("Depth", depth_colormap)
                # cv2.imshow("Color", color_image)

                # # Break the loop if 'q' is pressed
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                    # break

        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.client_socket.close()
            #cv2.destroyAllWindows()
    def receive_frame(self):
        try:
            
                # Receive the size of the incoming data
                size_data = self.client_socket.recv(4)
                # if not size_data:
                    # break
                size = int.from_bytes(size_data, byteorder='big')
                if(size>0):
                  # Receive the serialized data
                  serialized_data = b''
                  
                  while len(serialized_data) < size:
                    serialized_data += self.client_socket.recv(size - len(serialized_data))

                  # Deserialize the data
                  data = pickle.loads(serialized_data)
                  depth_image = data['depth']
                  color_image = data['color']
                  # Display the received frames
                  depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                  return color_image, depth_image
                else:
                  return None, None                
                # cv2.imshow("Depth", depth_colormap)
                # cv2.imshow("Color", color_image)

                # # Break the loop if 'q' is pressed
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                    # break

        except Exception as e:
            print(f"Error: {e}")

    