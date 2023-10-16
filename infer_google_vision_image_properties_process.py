import copy
from ikomia import core, dataprocess
from google.cloud import vision
import os
import io
import cv2
from PIL import Image, ImageDraw
import numpy as np


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferGoogleVisionImagePropertiesParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.google_application_credentials = ''

    def set_values(self, params):
        # Set parameters values from Ikomia Studio or API
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.window_size = int(params["window_size"])
        self.google_application_credentials = str(params["google_application_credentials"])

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {}
        params["google_application_credentials"] = str(self.google_application_credentials)
        return params


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferGoogleVisionImageProperties(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the algorithm here
        self.add_output(dataprocess.DataDictIO())
        self.add_output(dataprocess.CObjectDetectionIO())
        self.add_output(dataprocess.CImageIO())


        # Create parameters object
        if param is None:
            self.set_param_object(InferGoogleVisionImagePropertiesParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.client = None
        self.color = [0, 255, 255]
        self.total_width = 1200
        self.image_height = 800

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def run(self):
        # Main function of your algorithm
        self.begin_task_run()

        # Get input
        input = self.get_input(0)
        src_image = input.get_image()

        # Set output
        output_dict = self.get_output(1)
        output_box = self.get_output(2)
        self.forward_input_image(0, 3)
        output_box.init('img', 3)

        # Get parameters
        param = self.get_param_object()


        if self.client is None:
            if param.google_application_credentials:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = param.google_application_credentials
            self.client = vision.ImageAnnotatorClient()

        # Convert the NumPy array to a byte stream
        src_image = src_image[..., ::-1] # Convert to bgr
        is_success, image_buffer = cv2.imencode(".jpg", src_image)
        byte_stream = io.BytesIO(image_buffer)

        # Convert the byte stream to bytes
        image_bytes = byte_stream.getvalue()

        # Create an Image object for Google Vision API
        image = vision.Image(content=image_bytes)

        # Inference
        response = self.client.image_properties(image=image)
        props  = response.image_properties_annotation
        color_data = props.dominant_colors.colors

        # Create a blank image
        image = Image.new("RGB", (self.total_width, self.image_height))
        draw = ImageDraw.Draw(image)

        # The total pixel_fraction is not always 1. So it will be normalize
        pixel_fraction_total = sum(entry.pixel_fraction for entry in color_data)

        # Draw the color blocks
        x0 = 0
        for entry in color_data:
            # Extract color values
            red = int(entry.color.red)
            green = int(entry.color.green)
            blue = int(entry.color.blue)
            color_rgb = (red, green, blue)

            # Calculate the width of the block based on pixel_fraction
            block_width = int((entry.pixel_fraction / pixel_fraction_total) * self.total_width)
            x1 = x0 + block_width
            draw.rectangle([x0, 0, x1, self.image_height], fill=color_rgb)
            x0 = x1

        # Convert image to np array
        image_np = np.array(image)

        # Add image colormap to output 
        output = self.get_output(0)
        output.set_image(image_np)

        # Add dict to output
        output_dict.data = ({'image_properties_annotation': f'{response.image_properties_annotation}'})

        # Get box coordinates the analyzed area
        vertices_data = response.crop_hints_annotation.crop_hints[0].bounding_poly.vertices
        vertices = [(vertex.x,vertex.y) for vertex in vertices_data]
        x_box = vertices[0][0]
        y_box = vertices[0][1]
        w = vertices[1][0] - x_box
        h = vertices[2][0] - y_box

        # Add selected area graphics to output
        output_box.add_object(0, 'selected area', 1.0, float(x_box), float(y_box), float(w), float(h), self.color)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferGoogleVisionImagePropertiesFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_google_vision_image_properties"
        self.info.short_description = "Image Properties feature detects general attributes of the image, such as dominant color."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.icon_path = "images/cloud.png"
        self.info.path = "Plugins/Python/Other"
        self.info.version = "1.0.0"
        self.info.authors = "Google"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2023
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://cloud.google.com/vision/docs/detecting-properties"
        # Code source repository
        self.info.repository = "https://github.com/googleapis/python-vision"
        # Keywords used for search
        self.info.keywords = "Image properties,Dominant color,Cloud,Vision AI"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OTHER"

    def create(self, param=None):
        # Create algorithm object
        return InferGoogleVisionImageProperties(self.info.name, param)
