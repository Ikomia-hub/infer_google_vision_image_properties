from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_google_vision_image_properties.infer_google_vision_image_properties_process import InferGoogleVisionImagePropertiesFactory
        return InferGoogleVisionImagePropertiesFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_google_vision_image_properties.infer_google_vision_image_properties_widget import InferGoogleVisionImagePropertiesWidgetFactory
        return InferGoogleVisionImagePropertiesWidgetFactory()
