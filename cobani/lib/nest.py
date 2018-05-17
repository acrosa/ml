# import urllib2
import urllib.request as urllib2
import json
import time

nest_api_url = 'https://developer-api.nest.com'

def fetch_snapshot_url(token):
    headers = {
        'Authorization': "Bearer {0}".format(token),
    }
    req = urllib2.Request(nest_api_url, None, headers)
    response = urllib2.urlopen(req)
    data = json.loads(response.read())

    # Verify the account has devices
    if 'devices' not in data:
        raise APIError(error_result("Nest account has no devices"))
    devices = data["devices"]

     # Verify the account has cameras
    if 'cameras' not in devices:
        raise APIError(error_result("Nest account has no cameras"))
    cameras = devices["cameras"]

    # Verify the account has 1 Nest Cam
    if len(cameras.keys()) < 1:
        raise APIError(error_result("Nest account has no cameras"))

    camera_id = list(cameras.keys())[0]
    camera = cameras[camera_id]

    # Verify the Nest Cam has a Snapshot URL field
    if 'snapshot_url' not in camera:
        raise APIError(error_result("Camera has no snapshot URL"))
    snapshot_url = camera["snapshot_url"]

    return snapshot_url

def timestamp_filename():
    return str(time.time()) + '.jpg'

def download_image_from_url(url=None, filename=timestamp_filename(), destination_dir="images/all"):
  request = urllib2.Request(url)
  response = urllib2.urlopen(request)
  data = response.read()
  with open(destination_dir +"/"+ str(filename), "wb") as code:
    code.write(data)

def store_camera_image(token):
    try:
        print("Downloading image from camera...")
        camera_image_url = fetch_snapshot_url(token)
        download_image_from_url(camera_image_url)
        filename = timestamp_filename()
        print("downloaded image: " + filename)
    except Exception as inst:
         print("Exception trying to fetch camera image. Error: " + str(inst))

def fetch(config, repeat=-1):
    token = config.get("nest", "token")
    print("repeats every "+ str(repeat) + " seconds.")
    if repeat > 0:
        while(True):
            store_camera_image(token)
            time.sleep(repeat)
    else:
        store_camera_image(token)
