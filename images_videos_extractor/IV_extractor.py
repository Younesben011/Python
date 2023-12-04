import subprocess
import keyboard
import os
import sys
import time
from pathlib import Path
from threading import Thread
selected_item = 0
device_Ids = []
images_list1 = []
videos_list = []
image_count = 0
videos_count = 0


def key_handller(event):
    global selected_item
    # print(event.name)
    if (event.name != "enter"):
        clear_terminal()
    match(event.name):
        case "up":
            selected_item -= 1
            show_list(device_Ids)
        case "down":
            selected_item += 1
            show_list(device_Ids)
        case "esc":
            sys.exit(0)


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


def show_list(devices):
    global selected_item
    devices_num = len(devices)
    if (selected_item >= devices_num):
        selected_item = 0
    if (selected_item < 0):
        selected_item = devices_num-1
    for indx, device in enumerate(devices):
        if (selected_item == indx):
            print(f"> {device}")
        else:
            print(f"  {device}")


def devices_handller(stdout):

    # number of connected devices
    devices_count = 0
    devices_ids = []
    devices_list = stdout.split()
    devices_list = devices_list[4:]
    connected = True
    while connected:
        try:
            device_connected = devices_list.index("device")
            devices_ids.append(devices_list[0])
            devices_count += 1
            devices_list = devices_list[2:]
        except:
            connected = False
            return devices_ids


PATH = "sdcard"


def is_dir(Path):
    res = subprocess.run(
        f'adb shell [ -d "{Path}" ] && echo exists',
        shell=True,
        capture_output=True,
        text=True
    )
    output = res.stdout.strip()
    if (output == "exists"):
        return True
    return False


def list_Storage():
    global PATH
    global imgsPath
    global videosPath
    global image_count
    Str_phone_dirs = subprocess.run(f"adb shell ls {PATH}",
                                    capture_output=True,
                                    shell=True,
                                    text=True
                                    ).stdout
    phone_dirs = Str_phone_dirs.split()
    for dir in phone_dirs:
        if (dir != "Android"):
            PATH = os.path.join(PATH, dir)
            # print(os.path.isdir(PATH))
            PATH = str(PATH).replace("\\", "/")
            if (is_dir(PATH)):
                list_Storage()
            else:
                extension = Path(dir).suffix
                match(extensionCheck(extension)):
                    case "image":
                        image_count += 1
                        t1 = Thread(target=images_list_handller, args=(PATH,))
                        t1.start()
                        # images_list.append(PATH)
                    case "video":
                        t2 = Thread(target=videos_list_handller, args=(PATH,))
                        t2.start()
                PATH = os.path.dirname(PATH)
                PATH = str(PATH).replace("\\", "/")

    PATH = os.path.dirname(PATH)
    PATH = str(PATH).replace("\\", "/")

    # print("sssssss", PATH)


image_Count1 = 0

temp_list = []


def images_list_handller(PATH):
    print("image handller")
    global images_list1
    global image_Count1
    global temp_list
    image_Count1 += 1
    images_list1.append(PATH)
    extract_thread1 = Thread(
        target=extract_images)
    extract_thread2 = Thread(
        target=extract_images,)
    if (image_Count1 >= 40):
        if (extract_thread1.is_alive() or extract_thread2.is_alive()):
            temp_list = images_list1
            t2 = Thread(target=buzzy_extraction, args=(
                extract_thread1, extract_thread1, temp_list,))
            t2.start()
        else:
            extract_thread1 = Thread(
                target=extract_images, args=(images_list1[0:20],))
            extract_thread2 = Thread(
                target=extract_images, args=(images_list1[20:],))
            extract_thread1.start()
            extract_thread2.start()
        images_list1 = []
        image_Count1 = 0


def buzzy_extraction(extract_thread1, extract_thread2, temp_list):

    while (extract_thread1.is_alive or extract_thread2.is_alive):
        pass
    if (len(temp_list) > 0):
        extract_thread1 = Thread(
            target=extract_images, args=(temp_list[0:20],))
        extract_thread2 = Thread(
            target=extract_images, args=(temp_list[20:],))
        temp_list = []
        extract_thread1.start()
        extract_thread2.start()


def videos_list_handller(PATH):
    print("video handller")
    global videos_list
    global videos_count
    videos_count += 1
    videos_list.append(PATH)
    if (videos_count >= 40):
        videos_count = 0
        extract_thread1 = Thread(
            target=extract_videos, args=(videos_list[0:20],))
        extract_thread2 = Thread(
            target=extract_videos, args=(videos_list[20:],))
        extract_thread1.start()
        extract_thread2.start()
        videos_list = []


def extensionCheck(ext):
    if (ext == ".jpg" or
                ext == ".JPG" or
                ext == ".JPEG" or
                ext == ".jpeg" or
                ext == ".png" or
                ext == ".webp" or
                ext == ".PNG"
            ):
        return "image"
    elif (ext == ".mp4" or
            ext == ".3gpp" or
            ext == ".MP4" or
            ext == ".3GP" or
            ext == ".3gp"
          ):
        return "video"
    else:
        return "none"


device_status = subprocess.run("adb devices",
                               shell=True,
                               capture_output=True,
                               text=True)


def extract_images(images_list):
    global imgsPath
    print("Extract")
    print(len(images_list))
    imgsPath = str(imgsPath).replace("\\", "/")

    for image in images_list:
        print(image)
        extract = subprocess.run(f"adb pull {image} {imgsPath}",
                                 shell=True,
                                 capture_output=True,
                                 text=True)


def extract_videos(vidoes_list):
    global videosPath
    videosPath = str(videosPath).replace("\\", "/")
    for video in vidoes_list:
        print(video)
        extract = subprocess.run(f"adb pull {video} {videosPath}",
                                 shell=True,
                                 capture_output=True,
                                 text=True)


# Print the device status
# print(device_status.decode())
device_connected = device_status.stdout.find("device") >= 0
print()
print("--------------------------------")
device_Ids = devices_handller(device_status.stdout)
num_devices = len(device_Ids)
print(f"Number of the connected devices{num_devices}")
print(devices_handller(device_status.stdout))
clear_terminal()
# device_Ids = ["asdsa", "asdas", "adasd"]
if (num_devices > 1):
    show_list(device_Ids)
    keyboard.on_press(key_handller)
    keyboard.wait("enter")
# time.sleep(2)
deviceID = device_Ids[selected_item]
dirName = os.path.dirname(os.path.abspath(__file__))
imgsPath = os.path.join(dirName, f'{deviceID+"-images"}\\')
if not (os.path.isdir(imgsPath)):
    os.mkdir(imgsPath)
videosPath = os.path.join(dirName, f'{deviceID+"-videos"}\\')
if not (os.path.isdir(videosPath)):
    os.mkdir(videosPath)
print(imgsPath)
list_Storage()


imagesThread = Thread(target=extract_images)
vidoesThread = Thread(target=extract_videos)
