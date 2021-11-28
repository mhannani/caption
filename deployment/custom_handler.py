import json
import torch
import io
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from torchvision import transforms


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
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # with open("vocab.json") as f:
        #     self.vocabulary = json.load(f)

    def preprocess(self, request):
        request = request[0]
        image = request.get("data")
        if image is None:
            image = request.get("body")

        image = Image.open(io.BytesIO(image))
        transformed_image = self.transform(image)

        return transformed_image

    def inference(self, data):
        print("model", self.model)
        model_output = self.model.image_captioner(data)
        return model_output

    def postprocess(self, data):
        return data

    def handle(self, data, context):
        model_input = self.preprocess(data)
        # print("model input", model_input)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
