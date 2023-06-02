import cv2
import glob
import shutil
import os
import json
import random

def get_last_file_name(dir_):
    dir_ = dir_.replace("/","/")
    list_of_files = glob.glob(dir_+"/*") # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return "".join((char if char.isalnum() else " ") for char in latest_file).split()[-2]
    
def get_frame_from_vid (vid_file, frame_folder):
    video = cv2.VideoCapture(vid_file)
    i = int(get_last_file_name(frame_folder)[4:])
    os.makedirs(frame_folder, exist_ok=True)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        cv2.imshow("", frame)
        if cv2.waitKey(0) & 0xFF == ord('p'):
            i += 1
            cv2.imwrite(frame_folder+"/"+"ball"+str(i)+".jpg", frame)

# anylabeling (tl, br) to yolov5 annotation form (xc, yc, w_object, h_height)
def convert_json_2_yolo_format(json_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    classes = {'ball':0, 'basket':1, 'person':2}
    for json_file in os.listdir(json_folder):
        try:
            json_file = f'{json_folder}/{json_file}'
            with open(json_file, 'r') as f:
                data = json.load(f)
            h = data['imageHeight']
            w = data['imageWidth']
            p = data['imagePath'][:-4] + '.txt'
            for obj in data['shapes']:
                tl = obj['points'][0]
                br = obj['points'][1]
                label = obj['label']
                x = (tl[0] + br[0]) / (2 * w)
                y = (tl[1] + br[1]) / (2 * h)
                width = (br[0] - tl[0]) / w
                height = (br[1] - tl[1]) / h

                with open(f'{out_folder}/{p}', 'a') as f:
                    # Only convert 'ball', 'basket', 'person' label
                    try:
                        f.write(f'{classes[label]} {x} {y} {width} {height}\n')
                    # Ignore irrelevant labels
                    except:
                        pass
        except:
            print(json_file)

# Split data into training format folder
def train_test_split(X_folder, y_folder):
    os.makedirs('datasets')
    os.makedirs('datasets/train')
    os.makedirs('datasets/train/images')
    os.makedirs('datasets/train/labels')
    os.makedirs('datasets/val')
    os.makedirs('datasets/val/images')
    os.makedirs('datasets/val/labels')
    files = os.listdir(X_folder)
    random.shuffle(files)
    size_train = int(len(files)*0.8)
    size_test = int(len(files)*0.2)
    print(f'Train size: {size_train}')
    print(f'Test size: {size_test}')
    for i in files:
        if size_train==0:
            break
        src_file = os.path.join(X_folder, i)
        des_file = os.path.join('datasets/train/images', i)
        shutil.copy(src_file, des_file)

        label = i.split('.')[0]+'.txt'
        src_file = os.path.join(y_folder, label)
        des_file = os.path.join('datasets/train/labels', label)
        shutil.copy(src_file, des_file)
        size_train-=1

    for i in files:
        if i not in os.listdir('datasets/train/images'):
            src_file = os.path.join(X_folder, i)
            des_file = os.path.join('datasets/val/images', i)
            shutil.copy(src_file, des_file)

            label = i.split('.')[0]+'.txt'
            src_file = os.path.join(y_folder, label)
            des_file = os.path.join('datasets/val/labels', label)
            shutil.copy(src_file, des_file)
  

def rotate_image_generator(PATH):
    degs = [d for d in range(10, 110, 10)]
    for f in os.listdir(PATH):
        r_txt = os.path.join(PATH,f)
        img = cv2.imread(r_txt)
        name = f.split('.')

        with open(f'{r_txt.split(".")[0]}{r_txt.split(".")[-1]}.txt', 'w') as fp:
            pass

        for degree in degs:
            save = os.path.join(PATH,f'{name[0]}_{degree}')
            height, width = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), degree, 1)
            rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
            with open(f'{save}_{r_txt.split(".")[-1]}.txt', 'w') as fp:
                pass

            cv2.imwrite(f'{save}.{r_txt.split(".")[-1]}', rotated_img)

def move_files(src_folder, des_folder, file_format):
    os.makedirs(des_folder, exist_ok=True)
    for file in os.listdir(src_folder):
        if file.endswith(f'.{file_format}'):
            shutil.move(os.path.join(src_folder, file), os.path.join(des_folder, file))

if __name__ =='__main__':
    # Step 1: Get frames data from video
    #         get_frame_from_vid (r"D:\BasketBall\vids\ball11.mp4", 'ball_dataset')
    # Step 2: Label images using Anylabeling tool
    # Step 3: Move all json annotation to another folder
    #         move_files(r"D:\BasketBall\t1", r"D:\BasketBall\t2", 'txt') 
    # Step 4: Yolo annotation form
    #         convert_json_2_yolo_format('folder contains json annotations', 'folder name for containing yolo format annotation')
    # Step 5: Split data
    #         train_test_split('folder_images', 'folder_labels')
    # rotate_image_generator(PATH) - Only for background scene has no any obj
    pass

    