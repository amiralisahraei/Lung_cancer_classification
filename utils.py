import torch
from torchvision import transforms
from PIL import Image
from model_structure import PreTrainedClassificationModel


def transformer(image):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((224, 224), antialias=True),
        ]
    )
    return transform(image)


def model_pipeline(image):
    categories = [
        "adenocarcinoma",
        "large.cell.carcinoma",
        "normal",
        "squamous.cell.carcinoma",
    ]

    # model = torch.load("model.pth",map_location=torch.device('cpu'))
    model = PreTrainedClassificationModel(4)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, prediction = torch.max(output, 1)
        return categories[prediction]


if __name__ == "__main__":
    image_path = r"./adenocarcinoma.png"
    image = Image.open(image_path).convert("RGB")
    output = transformer(image)
    prediction = model_pipeline(output)
    print(prediction)
