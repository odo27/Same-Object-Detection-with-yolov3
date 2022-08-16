import numpy as np
import imutils
import time
import cv2
import os
import datetime
import sys
import pickle
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

file_directory = ["test.mp4"]
selected_file_directory = file_directory[-1]
selected_object_directory = []
stop_command = False
args = {'input':'test.mp4', 'output':'testout.avi', 'yolo':'yolo-coco', 'confidence':0.5, 'threshold':0.3}
c = open("result.txt", 'w')
c.close()
v = open("information.txt", 'w')
v.close()
allobjectinvideo = []
database_color = []
database_x = []
database_y = []
database_w = []
database_h = []
database_label = []
database_confidence = []
skip_frame_value = 0

class ShowVideo(QObject):

    VideoSignal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()

    def startVideo(self):
        global image

        capture = cv2.VideoCapture(selected_file_directory)

        ret, image = capture.read()
        height, width = image.shape[:2]

        while True:

            if stop_command == True:
                break
            
            if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
                capture.open(selected_file_directory)

            ret, image = capture.read()
            cimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            qimage = QImage(cimage.data, width, height, cimage.strides[0], QImage.Format_RGB888)
            
            self.VideoSignal.emit(qimage)



            loop = QEventLoop()
            QTimer.singleShot(25, loop.quit) #25 ms
            loop.exec_()


class ImageViewer(QWidget):

    def __init__(self):
        super().__init__()
        self.image = QImage()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image)
        self.image = QImage()

    
    def setImage(self, image):
        
        self.image = image
        
        if image.size() != self.size():
            self.setFixedSize(1080, 590)
        
        self.update()

class ThreadClass(QThread):

    def __init__(self):
        super().__init__()
        
    def run(self):
        global allobjectinvideo
        global database_color
        global database_x
        global database_y
        global database_w
        global database_h
        global database_label
        global database_confidence
        
        allobjectinvideo = []
        database_color = []
        database_x = []
        database_y = []
        database_w = []
        database_h = []
        database_label = []
        database_confidence = []

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
            dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
        configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()
        print(ln)
        print(net.getUnconnectedOutLayers())
        #ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        # initialize the video stream, pointer to output video file, and
        # frame dimensions
        vs = cv2.VideoCapture(args['input'])
        writer = None
        (W, H) = (None, None)

        # try to determine the total number of frames in the video file
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
            total = int(vs.get(prop))
            print("[INFO] {} total frames in video".format(total/(skip_frame_value+1)))

        # an error occurred while trying to determine the total
        # number of frames in the video file
        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            total = -1

        # loop over frames from the video file stream

        f = open("information.txt", 'w')
        framecount = 0

        while framecount < total:
            framecount += 1
            
            if framecount % (skip_frame_value+1) == 0:
                vs.set(cv2.CAP_PROP_POS_FRAMES, framecount)
                # if the frame was not grabbed, then we have reached the end
                # of the stream
                # read the next frame from the file
                (grabbed, frame) = vs.read()
                if not grabbed:
                    break
                # if the frame dimensions are empty, grab them
                if W is None or H is None:
                    (H, W) = frame.shape[:2]

                # construct a blob from the input frame and then perform a forward
                # pass of the YOLO object detector, giving us our bounding boxes
                # and associated probabilities
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                    swapRB=True, crop=False)
                net.setInput(blob)
                start = time.time()
                layerOutputs = net.forward(ln)
                end = time.time()

                # initialize our lists of detected bounding boxes, confidences,
                # and class IDs, respectively
                boxes = []
                confidences = []
                classIDs = []

                # loop over each of the layer outputs
                for output in layerOutputs:
                    # loop over each of the detections
                    for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > args["confidence"]:
                            # scale the bounding box coordinates back relative to
                            # the size of the image, keeping in mind that YOLO
                            # actually returns the center (x, y)-coordinates of
                            # the bounding box followed by the boxes' width and
                            # height
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top
                            # and and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates,
                            # confidences, and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)

                # apply non-maxima suppression to suppress weak, overlapping
                # bounding boxes
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                    args["threshold"])

                database_color_frame = []
                database_x_frame = []
                database_y_frame = []
                database_w_frame = []
                database_h_frame = []
                database_label_frame = []
                database_confidence_frame = []

                # ensure at least one detection exists
                if len(idxs) > 0:
                    f.write("\n")
                    f.write(str(framecount))
                    f.write(",")

                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        database_color_frame.append(COLORS[classIDs[i]])
                        database_x_frame.append(x)
                        database_y_frame.append(y)
                        database_w_frame.append(w)
                        database_h_frame.append(h)
                        database_label_frame.append(LABELS[classIDs[i]])
                        database_confidence_frame.append(confidences[i])

                        f.write(LABELS[classIDs[i]])
                        f.write(",")
                        f.write(str(x))
                        f.write(",")
                        f.write(str(y))
                        f.write(",")
                        f.write(str(w))
                        f.write(",")
                        f.write(str(h))
                        f.write(",")
                
                database_color.append(database_color_frame)
                database_x.append(database_x_frame)
                database_y.append(database_y_frame)
                database_w.append(database_w_frame)
                database_h.append(database_h_frame)
                database_label.append(database_label_frame)
                database_confidence.append(database_confidence_frame)

                # check if the video writer is None
                if writer is None:
                    writer = "end"
                    # some information on processing single frame
                    if total > 0:
                        elap = (end - start)
                        print("[INFO] single frame took {:.4f} seconds".format(elap))
                        print("[INFO] estimated total time to finish: {:.4f}".format(
                            elap * total / (skip_frame_value+1)))

        f.close()

        with open('data.pickle', 'wb') as f:
            pickle.dump(database_color, f)
            pickle.dump(database_x, f)
            pickle.dump(database_y, f)
            pickle.dump(database_w, f)
            pickle.dump(database_h, f)
            pickle.dump(database_label, f)
            pickle.dump(database_confidence, f)

        z = open("information.txt", 'r')

        zlines = z.readlines()

        #자료구조를 리스트형태로 변경한다
        m = 0
        while m < len(zlines):
            zlines[m] = zlines[m].split(",")
            m += 1

        #모든 프레임에 존재하는 객체들의 yolo 인식 결과값을 중복되지 않게 정리한다
        m = 1
        while m < len(zlines):
            n = 0
            while n < len(zlines[m])//5:
                if zlines[m][5*n+1] in allobjectinvideo:
                    pass
                else:
                    allobjectinvideo.append(zlines[m][5*n+1])
                n += 1
            m += 1
        
        z.close()
        
        # release the file pointers
        print("[INFO] cleaning up...")
        vs.release()

class ThreadClass2(QThread):
    def __init__(self):
        super().__init__()
    def run(self):
        vs = cv2.VideoCapture(args["input"])
        writer = None

        if database_x == []:
            reply = QMessageBox.question(QWidget(), 'Message', 'You have to run yolo first', QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                pass
        else:
            print('start making video')
            m = 0
            while m < len(database_x):
                vs.set(cv2.CAP_PROP_POS_FRAMES, (skip_frame_value+1)*(m+1))
                (grabbed, frame) = vs.read()

                if not grabbed:
                    break

                cv2.putText(frame, str((m+1)*(skip_frame_value+1)), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,0], 2)

                n = 0
                while n < len(database_x[m]):
                    if database_label[m][n] in selected_object_directory:
                        color = [int(c) for c in database_color[m][n]]
                        cv2.rectangle(frame, (database_x[m][n], database_y[m][n]), (database_x[m][n] + database_w[m][n], database_y[m][n] + database_h[m][n]), color, 2)
                        text = "{}: {:.4f}".format(database_label[m][n],
                            database_confidence[m][n])
                        cv2.putText(frame, text, (database_x[m][n], database_y[m][n] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        pass
                    n += 1
                for i in database_label[m]:
                    if i in selected_object_directory:
                        draw_frame = True
                        break
                    else:
                        draw_frame = False
                
                if writer is None:
                        # initialize our video writer
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                            (frame.shape[1], frame.shape[0]), True)
                    # write the output frame to disk
                
                if draw_frame:
                    writer.write(frame)
                m += 1
            print('finish making video')


def add_file_directory():
    global directory_layout

    globals()['radiobutton{}'.format(file_directory[-1])] = QRadioButton(file_directory[-1])
    globals()['radiobutton{}'.format(file_directory[-1])].clicked.connect(radiobutton_clicked)
    directory_layout.addWidget(globals()['radiobutton{}'.format(file_directory[-1])])

def add_video_button_clicked():
    global file_directory

    same_file_inspection = QFileDialog.getOpenFileName(main_window, 'Add Video', './')[0]
    if same_file_inspection == '':
        pass
    elif same_file_inspection in file_directory:
        reply = QMessageBox.question(QWidget(), 'Message', 'This file already exists', QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            pass
    else:
        file_directory.append(same_file_inspection)
        add_file_directory()

def delete_video_button_clicked():
    global file_directory

    for i in file_directory:
        if globals()['radiobutton{}'.format(i)].isChecked() == True:
            directory_layout.removeWidget(globals()['radiobutton{}'.format(i)])
            globals()['radiobutton{}'.format(i)].deleteLater()
            globals()['radiobutton{}'.format(i)] = None
            file_directory.remove(i)

def make_video_button_clicked():
    main_window.threadclass2.start()
    

def skip_frame_value_entered():
    global skip_frame_value
    try:
        skip_frame_value = int(skip_frame_value_le.text())
        skip_frame_value_label.setText('skip frame value : '+skip_frame_value_le.text())
    except:
        reply = QMessageBox.question(QWidget(), 'Message', 'You can type only integar', QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            pass

def print_result_button_clicked():
    result = open('result.txt').read()
    result_textedit = QTextEdit()
    result_textedit.setText(result)
    result_layout.addWidget(result_textedit, 0, 0)

def play_video_button_clicked():
    global stop_command

    stop_command = False

def stop_video_button_clicked():
    global stop_command

    stop_command = True

def run_yolo_button_clicked():
    main_window.threadclass.start()

def run_sod_button_clicked():
    f = open("information.txt", 'r')
    lines = f.readlines()
    objectcount = []
    database = []
    setstandard = 50
    with open('data.pickle', 'rb') as f:
        database_color = pickle.load(f)
        database_x = pickle.load(f)
        database_y = pickle.load(f)
        database_w = pickle.load(f)
        database_h = pickle.load(f)
        database_label = pickle.load(f)
        database_confidence = pickle.load(f)
    try:
        #자료구조를 리스트형태로 변경한다
        m = 0
        while m < len(lines):
            lines[m] = lines[m].split(",")
            m += 1

        #모든 프레임에 존재하는 객체들의 yolo 인식 결과값을 중복되지 않게 정리한다
        m = 1
        while m < len(lines):
            n = 0
            while n < len(lines[m])//5:
                if lines[m][5*n+1] in objectcount:
                    pass
                else:
                    objectcount.append(lines[m][5*n+1])
                n += 1
            m += 1

        #중복되지 않게 정리된 값과 0이란 초기 설정값으로 리스트를 구성한다
        m = 0
        while m < len(objectcount):
            objectcount[m] = [objectcount[m], 0]
            m += 1

        #프레임1에 나온 모든 객체를 데이터베이스에 추가한다
        m = 0
        while m < len(lines[1])//5:
            n = 0
            while n < len(objectcount):
                if lines[1][5*m+1] == objectcount[n][0]:
                    objectcount[n][1] += 1
                    database.append([objectcount[n][0]+str(objectcount[n][1]), '1'])
                n += 1
            m += 1
        
        #현재 프레임과 이전 프레임을 비교하여 동일 객체를 판별해 데이터베이스를 구축한다
        m = 2
        while m < len(lines):
            n = 0
            variableconstant = 0
            needtoremove = []
            while n < len(lines[m])//5:
                k = 0
                sameobjectproperty = [10000, 10000]
                while k < len(lines[m-1])//5:
                    if lines[m][5*n+1] == lines[m-1][5*k+1]:
                        centerx1 = float(lines[m-1][5*k+2]) + float(lines[m-1][5*k+4]) * 0.5
                        centery1 = float(lines[m-1][5*k+3]) + float(lines[m-1][5*k+5]) * 0.5
                        centerx2 = float(lines[m][5*n+2]) + float(lines[m][5*n+4]) * 0.5
                        centery2 = float(lines[m][5*n+3]) + float(lines[m][5*n+5]) * 0.5
                        distance = ( (centerx2 - centerx1) ** 2 + (centery2 - centery1) ** 2 ) ** 0.5
                        if distance < sameobjectproperty[0]:
                            sameobjectproperty[0] = distance
                            sameobjectproperty[1] = k
                    k += 1
                if sameobjectproperty[0] < setstandard:
                    sameobjectlocation = len(database) - k + sameobjectproperty[1] - variableconstant
                    database[sameobjectlocation].append(lines[m][0])
                    database.append(database[sameobjectlocation])
                    needtoremove.append(sameobjectlocation)
                    variableconstant += 1
                else:
                    l = 0
                    while l < len(objectcount):
                        if objectcount[l][0] == lines[m][5*n+1]:
                            objectcount[l][1] += 1
                            database.append([objectcount[l][0]+str(objectcount[l][1]), lines[m][0]])
                        l += 1
                    variableconstant += 1
                n += 1
            while True:
                if len(needtoremove) == 0:
                    break
                del database[needtoremove[0]]
                o = 1
                while o < len(needtoremove):
                    if needtoremove[o] > needtoremove[0]:
                        needtoremove[o] -= 1
                    o += 1
                del needtoremove[0]
            m += 1

        f.close()

        p = open("result.txt", 'w')

        m = 0
        while m < len(database):
            p.write(database[m][0])
            p.write("\n")
            p.write("   ")
            n = 1
            while n < len(database[m]):
                p.write(database[m][n])
                p.write(" ")
                n += 1
            p.write("\n")
            m += 1

        p.close()

        clearLayout(low_widget_layout)
        
        if allobjectinvideo == []:
            reply = QMessageBox.question(QWidget(), 'Message', 'You have to run yolo first', QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                pass

        m = 0
        while m < len(allobjectinvideo):
            globals()['radiobutton{}'.format(allobjectinvideo[m])] = QRadioButton(allobjectinvideo[m])
            globals()['radiobutton{}'.format(allobjectinvideo[m])].setAutoExclusive(False)
            globals()['radiobutton{}'.format(allobjectinvideo[m])].clicked.connect(object_radiobutton_clicked)
            low_widget_layout.addWidget(globals()['radiobutton{}'.format(allobjectinvideo[m])])
            m += 1
    
    except:
        reply = QMessageBox.question(QWidget(), 'Message', 'You have to run yolo first', QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            pass

def radiobutton_clicked():
    global selected_file_directory
    
    for i in file_directory:
        if globals()['radiobutton{}'.format(i)].isChecked() == True:
            selected_file_directory = globals()['radiobutton{}'.format(i)].text()
            args['input'] = selected_file_directory

def object_radiobutton_clicked():

    for i in allobjectinvideo:
        if globals()['radiobutton{}'.format(i)].isChecked() == True:
            if i in selected_object_directory:
                pass
            else:
                selected_object_directory.append(i)
        else:
            if i in selected_object_directory:
                selected_object_directory.remove(i)

def clearLayout(layout):
    while layout.count():
        child = layout.takeAt(0)
        if child.widget() is not None:
            child.widget().deleteLater()
        elif child.layout() is not None:
            clearLayout(child.layout())


if __name__ == '__main__':

    app = QApplication(sys.argv)

    main_window = QMainWindow()
    main_window.threadclass = ThreadClass()
    main_window.threadclass2 = ThreadClass2()
    
    
    show_video = ShowVideo()

    image_viewer = ImageViewer()
    show_video.VideoSignal.connect(image_viewer.setImage)

    add_video_action = QAction('Add Video')
    add_video_action.setShortcut('Ctrl+A')
    add_video_action.triggered.connect(add_video_button_clicked)
    delete_video_action = QAction('Delete Video')
    delete_video_action.setShortcut('Ctrl+D')
    delete_video_action.triggered.connect(delete_video_button_clicked)
    make_video_action = QAction('Make Video')
    make_video_action.setShortcut('Ctrl+W')
    make_video_action.triggered.connect(make_video_button_clicked)
    print_result_action = QAction('Print Result')
    print_result_action.setShortcut('Ctrl+I')
    print_result_action.triggered.connect(print_result_button_clicked)
    play_video_action = QAction('Play Video')
    play_video_action.setShortcut('Ctrl+P')
    play_video_action.triggered.connect(play_video_button_clicked)
    play_video_action.triggered.connect(show_video.startVideo)
    stop_video_action = QAction('Stop Yolo')
    stop_video_action.setShortcut('Ctrl+S')
    stop_video_action.triggered.connect(stop_video_button_clicked)
    run_yolo_action = QAction('Run Yolo')
    run_yolo_action.setShortcut('Ctrl+R')
    run_yolo_action.triggered.connect(run_yolo_button_clicked)
    run_sod_action = QAction('Run Sod')
    run_sod_action.setShortcut('Ctrl+O')
    run_sod_action.triggered.connect(run_sod_button_clicked)

    menubar = main_window.menuBar()
    menubar.setNativeMenuBar(False)

    fileMenu = menubar.addMenu('&File')
    fileMenu.addAction(add_video_action)
    fileMenu.addAction(delete_video_action)
    videoMenu = menubar.addMenu('&Video')
    videoMenu.addAction(play_video_action)
    videoMenu.addAction(stop_video_action)
    yoloMenu = menubar.addMenu('&Yolo')
    yoloMenu.addAction(run_yolo_action)
    yoloMenu.addAction(run_sod_action)
    opencvMenu = menubar.addMenu('&OpenCV')
    opencvMenu.addAction(make_video_action)
    resultMenu = menubar.addMenu('&Result')
    resultMenu.addAction(print_result_action)

    centralwidget = QWidget()
    main_window.setCentralWidget(centralwidget)

    option_group = QGroupBox('Option')

    add_video_button = QPushButton('&Add Video')
    add_video_button.clicked.connect(add_video_button_clicked)
    delete_video_button = QPushButton('&Delete Video')
    delete_video_button.clicked.connect(delete_video_button_clicked)
    make_video_button = QPushButton('&Make Video')
    make_video_button.clicked.connect(make_video_button_clicked)
    skip_frame_value_label = QLabel('skip frame value : ')
    skip_frame_value_le = QLineEdit()
    skip_frame_value_le.setMaximumWidth(500)
    skip_frame_value_le.returnPressed.connect(skip_frame_value_entered)
    print_result_button = QPushButton('Pr&int Result')
    print_result_button.clicked.connect(print_result_button_clicked)
    play_video_button = QPushButton('&Play Video')
    play_video_button.clicked.connect(play_video_button_clicked)
    play_video_button.clicked.connect(show_video.startVideo)
    stop_video_button = QPushButton('&Stop Video')
    stop_video_button.clicked.connect(stop_video_button_clicked)
    run_yolo_button = QPushButton('&Run Yolo')
    run_yolo_button.clicked.connect(run_yolo_button_clicked)
    run_sod_button = QPushButton('Run S&od')
    run_sod_button.clicked.connect(run_sod_button_clicked)

    directory_group = QGroupBox('Directory')
    globals()['radiobutton{}'.format(file_directory[-1])] = QRadioButton(file_directory[-1])
    globals()['radiobutton{}'.format(file_directory[-1])].setChecked(True)
    globals()['radiobutton{}'.format(file_directory[-1])].clicked.connect(radiobutton_clicked)

    video_group = QGroupBox('Video')

    result_group = QGroupBox('Result')

    high_widget = QWidget()
    high_widget_layout = QHBoxLayout()
    high_widget_layout.addWidget(add_video_button)
    high_widget_layout.addWidget(delete_video_button)
    high_widget.setLayout(high_widget_layout)
    middle_widget = QWidget()
    middle_widget_layout = QHBoxLayout()
    middle_widget_layout.addWidget(make_video_button)
    middle_widget_layout.addWidget(print_result_button)
    middle_widget.setLayout(middle_widget_layout)
    middle_2_widget = QWidget()
    middle_2_widget_layout = QHBoxLayout()
    middle_2_widget_layout.addWidget(skip_frame_value_label)
    middle_2_widget_layout.addWidget(skip_frame_value_le)
    middle_2_widget.setLayout(middle_2_widget_layout)
    low_widget = QWidget()
    low_widget_layout = QHBoxLayout()
    low_widget.setLayout(low_widget_layout)
    option_layout = QVBoxLayout()
    option_layout.addWidget(high_widget)
    option_layout.addWidget(middle_widget)
    option_layout.addWidget(middle_2_widget)
    option_layout.addWidget(low_widget)
    option_group.setLayout(option_layout)
    
    directory_layout = QVBoxLayout()
    directory_layout.addWidget(globals()['radiobutton{}'.format(file_directory[-1])])
    directory_group.setLayout(directory_layout)

    video_layout = QVBoxLayout()
    video_widget = QWidget()
    video_widget_layout = QHBoxLayout()
    video_widget_layout.addWidget(play_video_button)
    video_widget_layout.addWidget(stop_video_button)
    video_widget_layout.addWidget(run_yolo_button)
    video_widget_layout.addWidget(run_sod_button)
    video_widget.setLayout(video_widget_layout)
    video_layout.addWidget(image_viewer)
    video_layout.addWidget(video_widget)
    video_group.setLayout(video_layout)

    result_layout = QGridLayout()
    result_group.setLayout(result_layout)

    left_widget = QWidget()
    left_widget_layout = QVBoxLayout()
    left_widget_layout.addWidget(video_group)
    left_widget_layout.addWidget(directory_group)
    left_widget.setLayout(left_widget_layout)
    right_widget = QWidget()
    right_widget_layout = QVBoxLayout()
    right_widget_layout.addWidget(result_group)
    right_widget_layout.addWidget(option_group)
    right_widget.setLayout(right_widget_layout)
    layout = QHBoxLayout()
    layout.addWidget(left_widget)
    layout.addWidget(right_widget)
    centralwidget.setLayout(layout)

    main_window.setWindowTitle('Same Object Detection')
    main_window.setGeometry(0, 0, 1920, 1080)
    main_window.show()

    sys.exit(app.exec_())