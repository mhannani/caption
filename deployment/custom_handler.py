from ts.torch_handler.base_handler import BaseHandler


class CaptionHandler(BaseHandler):
    """
    A custom model handler implementation.
    The Handler takes a image as Tensor preprocess it
    and fed it to the network for inference.

    Then got a generated caption post-process it
    and sent it as a request response.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        """
        Get invoked by by torchserve for loading the model.
        :param context: context contains server system properties.
        :return: None
        """

        # load the model

