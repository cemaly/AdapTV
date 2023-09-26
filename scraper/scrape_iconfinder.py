import os
import random

import requests
from bs4 import BeautifulSoup

icon_labels = [
    "movie",
    "radio",
    "tv",
    "game",
    "picture",
    "video",
    "settings",
    "search",
    "grid",
    "power",
    "weather",
    "bluetooth",
    "wifi",
    "favourite",
    "file/folder",
    "youtube",
    "netflix",
    "play",
    "pause",
    "next",
    "stop",
    "rewind",
    "previous",
    "delete",
    "edit",
    "download",
    "upload",
    "input source",
    "browser",
    "recording",
    "add",
    "tools",
    "parental",
    "input scart",
    "HDMI",
    "music",
    "sound",
    "volume",
    "record",
    "screen share",
    "google",
    "amazon prime",
    "fast forward",
    "camera",
    "facebook",
    "twitter",
    "close",
    "gallery",
    "home",
    "microphone",
    "mute/unmute",
    "refresh",
    "sattelite",
    "save",
    "source",
    "spotify",
]

for label in icon_labels:

    query = f"{label} icon"
    url = f"https://www.flaticon.com/search?word={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.128 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    icons = soup.find_all("img", class_="lzy")

    # Create a directory to save the images
    os.makedirs(f"images/{label}", exist_ok=True)

    for icon in icons:
        image_url = icon["data-src"]
        response = requests.get(image_url)
        with open(f'images/{label}/' + str(random.randint(1, 2**65)) + ".png", "wb") as file:
            file.write(response.content)