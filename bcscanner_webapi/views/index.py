from django.shortcuts import render


def index(request):

    form_data = {
        "names": ["toto"],
        'titles': ["Mr."],
        'occupations': ["maker"],
        'organisations': [{"name": 'home', "acronym": "HM"}],
        'street': ['01 makers street'],
        'city': ['makerhood'],
        'country': ['makerstan'],
        'emails': ['toto@makers.org'],
        'www': ['www.totothemaker.com'],
        'phone_numbers': [{'type': 'home', 'number': '+123456789'}],
        'unassigned': ['blabla']
    }

    context = {
        "data": {
            "form_data": form_data,
            "default_form_data": form_data
        }
    }

    template = "bcscanner_webapi/index.html"

    return render(request, template, context=context)

