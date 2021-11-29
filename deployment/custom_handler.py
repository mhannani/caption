import json
import torch
import io
import os
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from torchvision import transforms
from models import Captioner


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
        with open("index_to_name.json") as f:
            self.vocabulary = json.load(f)

    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""
        # hyperparameters
        embed_size = 256
        hidden_size = 256
        vocabulary_size = 2339
        num_layer = 1

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        caption_checkpoint = torch.load(
                                os.path.join(model_dir, "checkpoint_num_39__21_11_2021__16_33_06.pth.tar"),
                                map_location=torch.device('cpu'))
        self.model = Captioner(embed_size, hidden_size, vocabulary_size, num_layer)
        self.model.load_state_dict(caption_checkpoint["state_dict"], strict=False)
        self.model.eval()

    def preprocess(self, request):
        print('request: ', request)
        request = request[0]
        print('request[0]: ', request)
        image = request.get("data")
        print('request.get(data): ', image)
        if image is None:
            image = request.get("body")

        image = Image.open(io.BytesIO(image))
        transformed_image = self.transform(image).unsqueeze(0)

        with open("index_to_name.json") as f:
            vocabulary = json.load(f)

        return transformed_image, vocabulary

    def inference(self, data, vocab):
        model_output = self.model.image_captioner(data, vocab)
        return model_output

    def postprocess(self, caption):
        json_result = []
        caption = " ".join(caption)

        # clean our caption string
        cleaned_caption = caption.replace('<EOS>', '').replace('<SOS>', '').strip()
        json_result.append({'caption': cleaned_caption})
        return json_result

    def handle(self, data, context):
        batch_image, vocabulary = self.preprocess(data)
        model_output = self.inference(batch_image, vocabulary)
        caption = self.postprocess(model_output)
        return caption
