from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_google_vision_image_properties.infer_google_vision_image_properties_process import InferGoogleVisionImagePropertiesParam

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferGoogleVisionImagePropertiesWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferGoogleVisionImagePropertiesParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)
        
        # Credentials
        self.browse_credentials = pyqtutils.append_browse_file(
                                            grid_layout=self.grid_layout,
                                            label="Google app credentials (.json)",
                                            path=self.parameters.google_application_credentials,
                                            mode=QFileDialog.ExistingFile
        )
        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        self.parameters.google_application_credentials = self.browse_credentials.path

        # Send signal to launch the algorithm main function
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferGoogleVisionImagePropertiesWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_google_vision_image_properties"

    def create(self, param):
        # Create widget object
        return InferGoogleVisionImagePropertiesWidget(param, None)
