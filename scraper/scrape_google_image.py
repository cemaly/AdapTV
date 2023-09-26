# Required imports
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import os


def download_images_for_label(label):
    """Download images for a given label from Google Images."""
    # Construct the search query and Google Images URL
    search_query = f'{label} wave icon'
    url = f"https://www.google.com/search?q={search_query}&tbm=isch"

    # Initialize a Chrome web driver
    driver = webdriver.Chrome(executable_path='chromedriver.exe')
    driver.get(url)

    # Scroll down multiple times to load more images on the webpage
    for _ in range(10):
        driver.execute_script("window.scrollBy(0,10000)")

    # Parse the page's content with BeautifulSoup
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    img_tags = soup.find_all("img", {"class": "rg_i"})

    # Ensure directory exists for the label
    os.makedirs(f"images/{label}", exist_ok=True)

    # Iterate over the image tags and download the image
    for i, img_tag in enumerate(img_tags):
        try:
            # Try to fetch image URL from 'src' attribute
            img_url = img_tag['src']
        except KeyError:
            # If 'src' is not present, fetch from 'data-src'
            img_url = img_tag.get('data-src', None)

        if img_url:
            # Download and save the image
            try:
                img_response = requests.get(img_url)
                with open(f'images/{label}/img{i}.png', 'wb') as f:
                    f.write(img_response.content)
            except Exception as e:
                print(f"Error downloading image {i} for label {label}. Error: {e}")

    # Close the Chrome web driver
    driver.close()


if __name__ == "__main__":
    # List of icon labels for which images are to be downloaded
    icon_labels = icon_labels = [
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

    # Currently set to only 'sound'. To download for all labels, 
    # comment this line and uncomment the complete list above.
    icon_labels = ['sound']

    for label in icon_labels:
        download_images_for_label(label)
