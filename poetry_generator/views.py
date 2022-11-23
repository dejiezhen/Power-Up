from django.shortcuts import render
from .poem import Poem

def index(request):
    """
    Render the frontend with backend data. Outputs poem into the frontend

    Args:
        none
    """
    print(type(request))
    context = {}
    if request.method == "POST":
        poem = Poem()
        speech_data = poem.speech_output(request)
        poem = poem.generate_poem(speech_data)
        context["poem"] = poem
    return render(request, './templates/home.html', context)