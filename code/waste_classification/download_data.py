import os, urllib.request, zipfile

URL = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
os.makedirs("data", exist_ok=True)
zip_path = "data/dataset-resized.zip"
if not os.path.exists(zip_path):
    print("Downloading TrashNet (~43 MB)...")
    urllib.request.urlretrieve(URL, zip_path)
with zipfile.ZipFile(zip_path) as z:
    z.extractall("data")
print("Done. Folders:", sorted(d for d in os.listdir("data/dataset-resized") if not d.startswith(".")))
