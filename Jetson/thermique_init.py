#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from uvctypes import *
import cv2
import numpy as np
import sys

from qimage2ndarray import array2qimage

from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# from PySide2 import QtCore
# from PySide2.QtGui import *
# from PySide2.QtWidgets import *
# from PySide2.QtCore import *

from datetime import datetime
import mysql.connector
from threading import Timer

threshold_value = 102.0
faceXY_1 = (0, 0)
faceXY_2 = (100, 100)


cnx = mysql.connector.connect(user='root', password='root',
                              host='13.126.201.239',
                              port=51784,
                              database='thermique')

myCursor = cnx.cursor()

try:
  from queue import Queue
except ImportError:
  from Queue import Queue
import platform
import paho.mqtt.client as mqtt


temperatureDHT = 0
fixed_point_X = -1
fixed_point_Y = -1
personTemperature = 0
personName = ""
previousRFID = ""

#------------------------------------------------------------------------------

def gstreamer_pipeline(
    capture_width=480,
    capture_height=270,
    display_width=480,
    display_height=270,
    framerate=20,
    flip_method=6,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class MainWindow(QWidget):
    keyPressedSignal = QtCore.pyqtSignal(str)

    def getPos(self, event):
      global fixed_point_X
      global fixed_point_Y

      fixed_point_X = event.pos().x()
      fixed_point_Y = event.pos().y()
      print("Updated Reference Point: [ ", event.pos().x(),", ", event.pos().y()," ]")

    def __init__(self):
        super(MainWindow, self).__init__()
#        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        

        self.HBL = QHBoxLayout()
        self.stringRead = ""
        # self.HBL.addStretch(1)
        self.LeftWidget = QWidget()
        self.LeftVBL = QVBoxLayout()
        self.RightVBL = QVBoxLayout()
        self.RGBCameraLabel = QLabel()
        self.ThermalCameraLabel = QLabel()
        self.ThermalCameraLabel.mousePressEvent = self.getPos

        self.freshScan = Timer(2.5, self.clearGreeting)

        self.keyPressedSignal.connect(self.on_rfid_scan)
        self.LeftVBL.addWidget(self.RGBCameraLabel)
        self.LeftVBL.addWidget(self.ThermalCameraLabel)

        self.LeftWidget.setLayout(self.LeftVBL)

        self.HBL.addWidget(self.LeftWidget)

        self.TemperatureLabel = QLabel()
        self.TemperatureLabel.setText("Please place your ID card\non the scanner")
        self.TemperatureLabel.setAlignment(Qt.AlignCenter)
        self.TemperatureLabel.setStyleSheet("color:white;\
                                            font-size:50px;")

        self.MessageLabel = QLabel()
        self.MessageLabel.setText("Make sure your face is visible\nwithin the blue box on the left")
        self.MessageLabel.setAlignment(Qt.AlignCenter)
        self.MessageLabel.setStyleSheet("color:yellow;\
                                          font-style: oblique;\
                                            font-size:45px;")

        self.RightVBL.addWidget(self.TemperatureLabel)
        self.RightVBL.addWidget(self.MessageLabel)

        self.RightWidget = QWidget()

        self.RightWidget.setLayout(self.RightVBL)

        self.HBL.addWidget(self.RightWidget)

        # self.CancelBTN = QPushButton("Cancel")
        # self.CancelBTN.clicked.connect(self.CancelFeed)
        # self.LeftVBL.addWidget(self.CancelBTN)

        self.Worker1 = Worker1()
        self.Worker2 = Worker2()

        self.Worker1.start()
        self.Worker2.start()

        self.Worker1.ImageUpdate.connect(self.RGBImageUpdateSlot)
        self.Worker2.ImageUpdate.connect(self.ThermalImageUpdateSlot)


        self.setStyleSheet("background-color: black;")
        self.setLayout(self.HBL)
        self.showFullScreen()

    def clearGreeting(self):
      self.TemperatureLabel.setText("Please place your ID card\non the scanner")
      self.TemperatureLabel.setAlignment(Qt.AlignCenter)
      self.TemperatureLabel.setStyleSheet("color:white;\
                                            font-size:50px;")

      self.MessageLabel.setText("Make sure your face is visible\nwithin the blue box on the left")
      self.MessageLabel.setAlignment(Qt.AlignCenter)
      self.MessageLabel.setStyleSheet("color:yellow;\
                                          font-style: oblique;\
                                            font-size:45px;")
      self.update()

    def keyPressEvent(self, event):
      if (event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return):
        # print("Enter key pressed")
        self.proceed()  
      elif (event.key() >= Qt.Key_0 or event.key() <= Qt.Key_Z):
        # print("Some other key...")
        self.stringRead += event.text()

    def getGreeting(self, rfid_value, temperature):
      myCursor.execute(f"SELECT name FROM people WHERE rfid='{rfid_value}'")
      global personName
      for name in myCursor:
        # print(name)
        personName = "" + name[0]
        break
      
      if(personName == ""):
        self.TemperatureLabel.setText(f"Your temperature is\n{temperature}")
        self.MessageLabel.setText("RFID card not found in records...")
        self.update()
        self.freshScan.cancel()
        self.freshScan = Timer(2.5, self.clearGreeting)
        self.freshScan.start()
        

      else:
        if(temperature > threshold_value):
          self.TemperatureLabel.setText(f"Your temperature is\n{temperature}")
          self.MessageLabel.setText(f"{personName}, you have a fever!\nPlease wait in quarantine.")
          self.MessageLabel.setStyleSheet("color:red;\
                                              font-size:50px;")
          self.update()
          self.freshScan.cancel()
          self.freshScan = Timer(2.5, self.clearGreeting)
          self.freshScan.start()
        else:
          self.TemperatureLabel.setText(f"Your temperature is\n{temperature}")
          self.MessageLabel.setText(f"Welcome back, {personName}.\nYou seem to be in good health!")
          self.MessageLabel.setStyleSheet("color:green;\
                                              font-size:50px;")
          self.update()
          self.freshScan.cancel()
          self.freshScan = Timer(2.5, self.clearGreeting)
          self.freshScan.start()

    def proceed(self):
        # RFID PROCESSING
        self.keyPressedSignal.emit(self.stringRead)
        rfid_value = ""
        rfid_value += self.stringRead
        print(self.stringRead)
        self.stringRead = ""
        self.processRFID(rfid_value)

    def on_rfid_scan(self, rfid_scanned):
        self.getGreeting(rfid_scanned, personTemperature)


    def RGBImageUpdateSlot(self, Image):
        self.RGBCameraLabel.setPixmap(QPixmap.fromImage(Image))

    def ThermalImageUpdateSlot(self, Image):
        self.ThermalCameraLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

    def processRFID(self, rfid_value):
      # DATABASE QUERIES HERE
      timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
      addRecord(myCursor, rfid_value, personTemperature, timestamp, 1)
      # UPDATE GREETING HERE AS WELL
      
def addRecord(myCursor, person_id, temperature, timestamp, gate):
  print('RFID:{0}\ntemperature:{1}\ntimestamp:{2}\ngate:{3}'.format(person_id, temperature, timestamp, gate))
  myCursor.execute(
      #f"SELECT person_id FROM `records` WHERE `person_id`={person_id} AND `date`='{date}' AND TIMEDIFF('{time}',`time`) <= '00:00:30'"
      f"SELECT rfid FROM `records` WHERE `rfid`={person_id} AND TIMESTAMPDIFF(SECOND,`timestamp`,'{timestamp}') <= '30' AND TIMESTAMPDIFF(SECOND,`timestamp`,'{timestamp}') > '0'"
      #this ^ line either returns a single row, otherwise empty row
  )
  redundant=myCursor.fetchall()
  #if the SELECT query returns an empty row, variable "redundant" is set to [], or None, which is logically equivalent to False.
  #otherwise "redundant" will store [(person_id,)]
  #yep, it's a tuple inside an array ^
  
  
  if(not redundant):
      myCursor.execute(
          #f"INSERT INTO `records` VALUES ('{person_id}', '{temperature}', '{time}', '{date}', '{gate}')""{0:.1f} degF".format(val)
          f"INSERT INTO `records` (`created_at`, `updated_at`, `rfid`, `temperature`, `timestamp`, `gate`) VALUES (CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,'{person_id}', '{temperature}', '{timestamp}', '{gate}')"
      )
  else:
      myCursor.execute(
          #f"UPDATE `records` SET `temperature`='{temperature}', `time`='{time}' WHERE `person_id`={person_id} AND `date`='{date}' AND TIMEDIFF('{time}',`time`) <= '00:00:30'"
          f"UPDATE `records` SET `updated_at`=CURRENT_TIMESTAMP, `temperature`='{temperature}', `timestamp`='{timestamp}' WHERE `rfid`={person_id} AND TIMESTAMPDIFF(SECOND,`timestamp`,'{timestamp}') <= '30' AND TIMESTAMPDIFF(SECOND,`timestamp`,'{timestamp}') > '30'"
      ) 

  myCursor.execute("COMMIT")


class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def run(self):

        self.face_detect()
        # self.ThreadActive = True


        # TODO: Declare proper video capture

        # Capture = cv2.VideoCapture(
        #     "rtsp://127.0.0.1:8001")
        # while self.ThreadActive:


        # # TODO: CUDNN FACE DETECT + Bounding Box

        #     ret, frame = Capture.read()
        #     if ret:
        #         Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         FlippedImage = cv2.flip(Image, 1)
        #         ConvertToQtFormat = QImage(
        #             FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
        #         Pic = ConvertToQtFormat.scaled(800, 480, Qt.KeepAspectRatio)
        #         self.ImageUpdate.emit(Pic)

    def stop(self):
        self.ThreadActive = False
        self.quit()

    def face_detect(self):
        face_cascade = cv2.CascadeClassifier(
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        )
        eye_cascade = cv2.CascadeClassifier(
            "/usr/share/opencv4/haarcascades/haarcascade_eye.xml"
        )
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        # cap = cv2.VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")
        global faceXY_1
        global faceXY_2
        if cap.isOpened():
            # cv2.namedWindow("Face Detect", cv2.WINDOW_AUTOSIZE)
            while True:
                ret, img = cap.read()
                img = cv2.resize(img[0:269, 20:359], (360, 270))
                # print("________________________________")
                # print(type(img))
                # print("________________________________")
                # img = img[0:270, 0:360]

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    faceXY_1 = (int(x*2), int(y*2))
                    faceXY_2 = (int((x+w)*2), int((y+h)*2))

                    roi_gray = gray[y : y + h, x : x + w]
                    roi_color = img[y : y + h, x : x + w]

                    #detect first face, and take in next frame
                    break


                # print("______________RGB HERE______________")
                # print(type(img))
                # print(type(img.data))
                # print("____________________________________")


                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                Pic = array2qimage(img)
                Pic = Pic.scaled(720, 540, Qt.IgnoreAspectRatio)
                self.ImageUpdate.emit(Pic)                    

                    #eyes = eye_cascade.detectMultiScale(roi_gray)
                    #for (ex, ey, ew, eh) in eyes:
                    #    cv2.rectangle(
                    #        roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                    #    )

                # cv2.imshow("Face Detect", img)
                # keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                # if keyCode == 27:
                    # break

            # cv2.destroyAllWindows()
        else:
            print("Unable to open camera")


#------------------------------------------------------------------------------


class Worker2(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def run(self):
        main(self)
        
    def stop(self):
        self.quit()
#------------------------------------------------------------------------------
  
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("temperatureDHT")
    client.loop_start()


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
  global temperatureDHT
  try:
    temperatureDHT = float(msg.payload)
  except ValueError:
    print("Couldn't convert to float")
    temperatureDHT = 0
  # print("TemperatureDHT: ", str(msg.payload))
    
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set("mqtt", password="thermique")
try:
  client.connect("192.168.1.170", 1883, 60)
except:
  pass
# client.loop_forever()

BUF_SIZE = 2
q = Queue(BUF_SIZE)

def numpyQImage(image):
    qImg = QImage()
    if image.dtype == np.uint8:
        if len(image.shape) == 2:
            channels = 1
            height, width = image.shape
            bytesPerLine = channels * width
            qImg = QImage(
                image.data, width, height, bytesPerLine, QImage.Format_Indexed8
            )
            qImg.setColorTable([qRgb(i, i, i) for i in range(256)])
        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                height, width, channels = image.shape
                bytesPerLine = channels * width
                qImg = QImage(
                    image.data, width, height, bytesPerLine, QImage.Format_RGB888
                )

def py_frame_callback(frame, userptr):

  array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
  data = np.frombuffer(
    array_pointer.contents, dtype=np.dtype(np.uint16)
  ).reshape(
    frame.contents.height, frame.contents.width
  ) # no copy

  # data = np.fromiter(
  #   frame.contents.data, dtype=np.dtype(np.uint8), count=frame.contents.data_bytes
  # ).reshape(
  #   frame.contents.height, frame.contents.width, 2
  # ) # copy

  if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
    return

  if not q.full():
    q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

def ktof(val):
  return (1.8 * ktoc(val) + 32.0)

def ktoc(val):
  return (val - 27315) / 100.0

def raw_to_8bit(data):
  cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
  np.right_shift(data, 8, data)
  return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

def display_temperature(img, val_k, loc, color):
  val = ktof(val_k)
  cv2.putText(img,"{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
  x, y = loc
  cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
  cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

def main(thread):
  ctx = POINTER(uvc_context)()
  dev = POINTER(uvc_device)()
  devh = POINTER(uvc_device_handle)()
  ctrl = uvc_stream_ctrl()

  res = libuvc.uvc_init(byref(ctx), 0)
  if res < 0:
    print("uvc_init error")
    exit(1)

  try:
    res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
    if res < 0:
      print("uvc_find_device error")
      exit(1)

    try:
      res = libuvc.uvc_open(dev, byref(devh))
      if res < 0:
        print("uvc_open error")
        exit(1)

      print("device opened!")

      print_device_info(devh)
      print_device_formats(devh)

      frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
      if len(frame_formats) == 0:
        print("device does not support Y16")
        exit(1)

      libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
        frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
      )

      res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
      if res < 0:
        print("uvc_start_streaming failed: {0}".format(res))
        exit(1)
      try:
        # client.loop()
        while True:
          data = q.get(True, 500)
          if data is None:
            break
          # print( tuple(data.shape[1::-1]))
          data = cv2.flip(data, 1)
          data = cv2.resize(data[0:83, 0:111], (720, 540))

          # TODO: TAKE FACIAL REGION FROM ANOTHER PLACE
          # ----------------------------------------------------------------------------------------
          global faceXY_1
          global faceXY_2
          #-----------------------------------------------------------------------------------------

          face_box = data[faceXY_1[1]:faceXY_2[1], faceXY_1[0]:faceXY_2[0]]
          global personTemperature
          minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(face_box)
          img = raw_to_8bit(data)
          cv2.rectangle(img,faceXY_1, faceXY_2,(0,255,255),2)

          # minLoc2 = (faceXY_1[0] + minLoc[0], faceXY_1[1] + minLoc[1])
          maxLoc2 = (faceXY_1[0] + maxLoc[0], faceXY_1[1] + maxLoc[1])

          # print("MinLoc: ", minLoc2)
          # print("MaxLoc: ", maxLoc2)

          # display_temperature(img, minVal, minLoc2, (255, 0, 0))
          display_temperature(img, maxVal, maxLoc2, (0, 0, 255))

          # BIAS AND EXTRACT

          if(temperatureDHT != 0 and fixed_point_X != -1 and fixed_point_Y != -1):
            print("DHT temperature found, ", temperatureDHT)
            bias = data[fixed_point_X, fixed_point_Y] - temperatureDHT
            print("bias: ", bias)
            maxVal = maxVal - bias
            print("maxVal: ", maxVal)

          personTemperature = float("{:.2f}".format(ktof(maxVal)))
          # print("MAXVAL: ", ktof(maxVal))


          # cv2.imshow('Lepton Radiometry', img)
          # cv2.setMouseCallback('Lepton Radiometry', click_event)
          # print("__________THERMALS HERE_____________")
          # print(type(img))
          # print(type(img.data))
          # print("____________________________________")

          
          
          Pic = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)

          # print("TYPE: ", type(Pic))
          thread.ImageUpdate.emit(Pic)
          
          # cv2.waitKey()
          # print(temperatureDHT)
          
        # cv2.destroyAllWindows()

      finally:
        libuvc.uvc_stop_streaming(devh)

      print("done")
    finally:
      libuvc.uvc_unref_device(dev)
  finally:
    libuvc.uvc_exit(ctx)
    client.loop_stop()
    cnx.close()


if __name__ == '__main__':
  App = QApplication(sys.argv)
  Root = MainWindow()
  Root.show()
  sys.exit(App.exec())


