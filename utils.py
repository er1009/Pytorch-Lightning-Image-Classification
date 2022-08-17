import timm
import argparse


def create_model(model="efficientnet_b3", num_classes=1, pretrained=True):
    """Download and builds a classification model

    Args:
        model (str, optional): model architecture such as resnet50_. Defaults to "efficientnet_b3".
        num_classes (int, optional): Num classes for training. Defaults to 1.
        pretrained (bool, optional): A flag wether we want a pretrained model or not. Defaults to True.
    Returns:
        model
    """
    # Check if model is supported
    model_name = model.lower()
    avail_pretrained_models = timm.list_models(pretrained=pretrained)
    if model_name.lower() not in avail_pretrained_models:
        raise AttributeError

    if pretrained:
        model = timm.create_model(model_name, pretrained)
        model.reset_classifier(num_classes=num_classes)

    else:
        model = timm.create_model(model_name, num_classes=num_classes)

    # unfreeze all layers
    for _, param in model.named_parameters():
        param.requires_grad = True

    return model

def load_yaml(path):
    import yaml

    with open(path, 'r') as f:
        y = yaml.load(f, yaml.SafeLoader)
    return y

def create_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--conf_path', required=True, help='path to the configuration file')
    return parser.parse_args()