# Copyright Philip Morris Products S.A. 2019

import sys
import os

from django.http import JsonResponse, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt

try:
    import bcscanner
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
    import bcscanner


@csrf_exempt
def scan_card(request):

    if request.method == 'POST':

        scanner = bcscanner.BCScanner()

        results = {}
        for key in request.FILES:
            image_file = request.FILES[key]
            image_type = image_file.content_type

            results[key] = scanner.scan_image(image_file.read(), image_type, images_as_data_urls=True)

        return JsonResponse(results)

    return HttpResponseForbidden()
