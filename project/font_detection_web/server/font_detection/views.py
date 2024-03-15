import base64
from django.shortcuts import render
from .forms import ImageUploadForm


def get_prediction(image_bytes):
    return "其他"


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
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)

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
