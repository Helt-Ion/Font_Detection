import io
import base64
from PIL import Image
from django.shortcuts import render
from .forms import ImageUploadForm
from .src.font_recognize import init, recognize_font_pil

model, classes_dict = init("font_detection/checkpoint", "font_recognize_100.pth")


def get_prediction(image_bytes):
    pil_image = Image.open(io.BytesIO(image_bytes))
    font_type = recognize_font_pil(pil_image, model, classes_dict)
    return font_type


def index(request):
    image_uri = None
    predicted_label = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # passing the image as base64 string to avoid storing it to DB or filesystem
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = f'data:image/jpeg;base64,{encoded_img}'

            # get predicted label
            try:
                predicted_label = get_prediction(image_bytes)
            except RuntimeError as re:
                print(re)
                # predicted_label = "Prediction Error"

    else:
        form = ImageUploadForm()

    context = {
        'form': form,
        'image_uri': image_uri,
        'predicted_label': predicted_label,
    }
    return render(request, 'index.html', context)
