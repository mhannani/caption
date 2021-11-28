from custom_handler import MyHandler

_service = MyHandler()


def handle(data, context):
    """
    Handle inference requests to torchserve.
    :param data: supplied data
    :param context: the context of server
    :return: predicted caption
    """

    # if the handler not yet initialized with the context
    if not _service.initialized:
        _service.initialize(context)

    # Not data supplied
    if data is None:
        return None

    # preprocess, inference and postprocess our image
    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
