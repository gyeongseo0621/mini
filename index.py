# -*- coding: utf-8 -*-

import platform
import numpy as np
import argparse
import cv2
import serial
import time
import sys
from threading import Thread

import math

X_255_point = 0
Y_255_point = 0
X_Size = 0
Y_Size = 0
Area = 0
Angle = 0
# -----------------------------------------------
Top_name = 'Kongdols Team Setting'
hsv_Lower = 0
hsv_Upper = 0

hsv_Lower0 = 0
hsv_Upper0 = 0

hsv_Lower1 = 0
hsv_Upper1 = 0

# -----------  0:노란색, 1:빨강색, 3:파란색
color_num = [0, 1, 2, 3, 4]

h_max = [107, 65, 196, 111, 110]
h_min = [77, 0, 158, 59, 74]

s_max = [164, 200, 223, 110, 255]
s_min = [139, 140, 150, 51, 133]

v_max = [82, 151, 239, 156, 255]
v_min = [63, 95, 104, 61, 104]

min_area = [50, 50, 50, 10, 10]

now_color = 0
serial_use = 1

serial_port = None
Temp_count = 0
Read_RX = 0

mx, my = 0, 0

threading_Time = 5 / 1000.


# -----------------------------------------------

def nothing(x):
    pass


# 트랙바를 조정할 때 마다 실행되는 콜백 함수를 정의해야 합니다.
# 포스팅의 예제에선 트랙바를 조절할 때마다 따로 실행할 명령이 없기 때문에
# 아무일도 하지 않는 더미 함수를 만듭니다.

# -----------------------------------------------
def create_blank(width, height, rgb_color=(0, 0, 0)):
    image = np.zeros((height, width, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color

    return image


# -----------------------------------------------
def draw_str2(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), lineType=cv2.LINE_AA)


# -----------------------------------------------
def draw_str3(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), lineType=cv2.LINE_AA)


# -----------------------------------------------
def draw_str_height(dst, target, s, height):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, height, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, height, (255, 255, 255), lineType=cv2.LINE_AA)


# -----------------------------------------------
def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


# -----------------------------------------------

def Trackbar_change(now_color):
    global hsv_Lower, hsv_Upper
    hsv_Lower = (h_min[now_color], s_min[now_color], v_min[now_color])
    hsv_Upper = (h_max[now_color], s_max[now_color], v_max[now_color])


# -----------------------------------------------
def Hmax_change(a):
    h_max[now_color] = cv2.getTrackbarPos('Hmax', Top_name)
    Trackbar_change(now_color)


# 트랙바에 Hmax 라 이름을 주고 Top_name 윈도우(?) 접근;
# -----------------------------------------------
def Hmin_change(a):
    h_min[now_color] = cv2.getTrackbarPos('Hmin', Top_name)
    Trackbar_change(now_color)


# -----------------------------------------------
def Smax_change(a):
    s_max[now_color] = cv2.getTrackbarPos('Smax', Top_name)
    Trackbar_change(now_color)


# -----------------------------------------------
def Smin_change(a):
    s_min[now_color] = cv2.getTrackbarPos('Smin', Top_name)
    Trackbar_change(now_color)


# -----------------------------------------------
def Vmax_change(a):
    v_max[now_color] = cv2.getTrackbarPos('Vmax', Top_name)
    Trackbar_change(now_color)


# -----------------------------------------------
def Vmin_change(a):
    v_min[now_color] = cv2.getTrackbarPos('Vmin', Top_name)
    Trackbar_change(now_color)


# -----------------------------------------------
def min_area_change(a):
    min_area[now_color] = cv2.getTrackbarPos('Min_Area', Top_name)
    if min_area[now_color] == 0:
        min_area[now_color] = 1
        cv2.setTrackbarPos('Min_Area', Top_name, min_area[now_color])
    Trackbar_change(now_color)


# -----------------------------------------------
def Color_num_change(a):
    global now_color, hsv_Lower, hsv_Upper
    now_color = cv2.getTrackbarPos('Color_num', Top_name)
    cv2.setTrackbarPos('Hmax', Top_name, h_max[now_color])
    cv2.setTrackbarPos('Hmin', Top_name, h_min[now_color])
    cv2.setTrackbarPos('Smax', Top_name, s_max[now_color])
    cv2.setTrackbarPos('Smin', Top_name, s_min[now_color])
    cv2.setTrackbarPos('Vmax', Top_name, v_max[now_color])
    cv2.setTrackbarPos('Vmin', Top_name, v_min[now_color])
    cv2.setTrackbarPos('Min_Area', Top_name, min_area[now_color])

    hsv_Lower = (h_min[now_color], s_min[now_color], v_min[now_color])
    hsv_Upper = (h_max[now_color], s_max[now_color], v_max[now_color])


# cv2.setTrackbarPos( 내가 설정한 이름, 윈도우 이름  , 초기값 )
# -----------------------------------------------
def TX_data(serial, one_byte):  # one_byte= 0~255
    global Temp_count
    try:
        serial.write(chr(int(one_byte)))
    except:
        Temp_count = Temp_count + 1
        print("Serial Not Open " + str(Temp_count))
        pass


# -----------------------------------------------
def RX_data(serial):
    global Temp_count
    try:
        if serial.inWaiting() > 0:
            result = serial.read(1)
            RX = ord(result)
            return RX
        else:
            return 0
    except:
        Temp_count = Temp_count + 1
        print("Serial Not Open " + str(Temp_count))
        return 0
        pass


# -----------------------------------------------

# *************************
# mouse callback function
def mouse_move(event, x, y, flags, param):
    global mx, my

    if event == cv2.EVENT_MOUSEMOVE:
        mx, my = x, y


# *************************
def receiving(ser):
    global receiving_exit

    global X_255_point
    global Y_255_point
    global X_Size
    global Y_Size
    global Area, Angle

    receiving_exit = 1
    while True:
        if receiving_exit == 0:
            break
        time.sleep(threading_Time)
        while ser.inWaiting() > 0:
            result = ser.read(1)
            RX = ord(result)
            # print ("RX=" + str(RX))
            if RX >= 100 and RX < 200:  # Color mode
                now_color = (RX - 100) / 10
                cv2.setTrackbarPos('Color_num', Top_name, now_color)
                RX = RX % 10
                if RX == 2:  # Center - X
                    ser.write(chr(int(X_255_point)))
                elif RX == 3:  # Center - Y
                    ser.write(chr(int(Y_255_point)))
                elif RX == 4:  # X_Size
                    ser.write(chr(int(X_Size)))
                elif RX == 5:  # Y_Size
                    ser.write(chr(int(Y_Size)))
                elif RX == 6:  # Angle
                    ser.write(chr(int(Angle)))
                    print("106=>" + str(Angle))
                elif RX == 1:  # Area
                    ser.write(chr(int(Area)))
                else:
                    ser.write(chr(int(0)))

            else:
                ser.write(chr(int(0)))


def GetLengthTwoPoints(XY_Point1, XY_Point2):
    return math.sqrt((XY_Point2[0] - XY_Point1[0]) ** 2 + (XY_Point2[1] - XY_Point1[1]) ** 2)


# *************************
def FYtand(dec_val_v, dec_val_h):
    return (math.atan2(dec_val_v, dec_val_y) * (180.0 / math.pi))


# *************************
# degree 값을 라디안 값으로 변환하는 함수
def FYrtd(rad_val):
    return (rad_val * (180.0 / math.pi))


# *************************
# 라디안값을 degree 값으로 변환하는 함수
def FYdtr(dec_val):
    return (dec_val / 180.0 * math.pi)


# *************************
def GetAngleTwoPoints(XY_Point1, XY_Point2):
    xDiff = XY_Point2[0] - XY_Point1[0]
    yDiff = XY_Point2[1] - XY_Point1[1]
    cal = math.degrees(math.atan2(yDiff, xDiff)) + 90
    if cal > 90:
        cal = cal - 180
    return cal


# *************************
# **************************************************
# **************************************************
# **************************************************
if __name__ == '__main__':

    # -------------------------------------
    print("-------------------------------------")
    print("[2019-11-22] MINI Robot Program.    Kongdols Corp.")
    print("-------------------------------------")
    print("")
    os_version = platform.platform()
    print(" ---> OS " + os_version)
    python_version = ".".join(map(str, sys.version_info[:3]))
    print(" ---> Python " + python_version)
    opencv_version = cv2.__version__
    print(" ---> OpenCV  " + opencv_version)

    # -------------------------------------
    # ---- user Setting -------------------
    # -------------------------------------
    W_View_size = 320
    # H_View_size = int(W_View_size / 1.777)
    H_View_size = int(W_View_size / 1.333)

    BPS = 4800  # 4800,9600,14400, 19200,28800, 57600, 115200
    serial_use = 1  ###### <+=========== ************* ADFADSFASFSD
    now_color = 0
    View_select = 1
    # -------------------------------------
    print(" ---> Camera View: " + str(W_View_size) + " x " + str(H_View_size))
    print("")
    print("-------------------------------------")
    # -------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64,
                    help="max buffer size")
    args = vars(ap.parse_args())

    img = create_blank(320, 50, rgb_color=(0, 0, 255))

    cv2.namedWindow(Top_name)

    cv2.createTrackbar('Hmax', Top_name, h_max[now_color], 255, Hmax_change)
    cv2.createTrackbar('Hmin', Top_name, h_min[now_color], 255, Hmin_change)
    cv2.createTrackbar('Smax', Top_name, s_max[now_color], 255, Smax_change)
    cv2.createTrackbar('Smin', Top_name, s_min[now_color], 255, Smin_change)
    cv2.createTrackbar('Vmax', Top_name, v_max[now_color], 255, Vmax_change)
    cv2.createTrackbar('Vmin', Top_name, v_min[now_color], 255, Vmin_change)
    cv2.createTrackbar('Min_Area', Top_name, min_area[now_color], 255, min_area_change)
    cv2.createTrackbar('Color_num', Top_name, color_num[now_color], 4, Color_num_change)

    Trackbar_change(now_color)

    draw_str3(img, (15, 25), 'Kongdols Corp.')

    cv2.imshow(Top_name, img)
    # ---------------------------
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])
    # ---------------------------
    camera.set(3, W_View_size)
    camera.set(4, H_View_size)

    time.sleep(0.5)
    # ---------------------------

    if serial_use != 0:
        serial_port = serial.Serial('/dev/ttyAMA0', BPS, timeout=0.001)
        serial_port.flush()  # serial cls

    # ---------------------------
    (grabbed, frame) = camera.read()
    draw_str2(frame, (5, 15), 'X_Center x Y_Center =  Area')
    draw_str2(frame, (5, H_View_size - 5), 'View: %.1d x %.1d.  Space: Fast <=> Video and Mask.'
              % (W_View_size, H_View_size))
    draw_str_height(frame, (5, int(H_View_size / 2)), 'Fast operation...', 3.0)
    cv2.imshow('Kongdols(frame) - Video', frame)

    cv2.setMouseCallback('Kongdols(frame) - Video', mouse_move)
    #
    # ---------------------------

    if serial_use != 0:
        t = Thread(target=receiving, args=(serial_port,))
        time.sleep(0.1)
        t.start()
    #
    # First -> Start Code Send
    # TX_data(serial_port, 1)
    # TX_data(serial_port, 1)
    # TX_data(serial_port, 1)

    old_time = clock()

    # -------- Main Loop Start --------
    while True:
        key = cv2.waitKey(1)  # key==ord("a")
        if key == 27:
            break

        # grab the current frame
        (grabbed, frame) = camera.read()

        if not grabbed:
            break

        height = frame.shape[0]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        # cut_frame = frame[2 * height // 3:, :]
        # cut_hsv = hsv[2 * height // 3:, :]  # HSV => YUV
        # frame[행시작:행끝, 열시작:열끝]

        cut_frame1 = frame[2 * height // 3 : , :2 * width // 3]
        cut_frame2 = frame[2 * height // 3 : , 1 * width // 3 : 2 * width // 3]
        cut_frame3 = frame[2 * height // 3 :, 2 * width // 3 : ]
        cut_hsv1 = hsv[2 * height // 3 : , :2 * width // 3]  # HSV => YUV
        cut_hsv2 = hsv[2 * height // 3 : , 1 * width // 3 : 2 * width // 3]
        cut_hsv3 = hsv[2 * height // 3 : , 2 * width // 3 : ]


        mask1 = cv2.inRange(cut_hsv1, hsv_Lower, hsv_Upper)
        mask1 = cv2.erode(mask1, None, iterations=1)
        mask1 = cv2.dilate(mask1, None, iterations=1)
        mask2 = cv2.inRange(cut_hsv2, hsv_Lower, hsv_Upper)
        mask2 = cv2.erode(mask2, None, iterations=1)
        mask2 = cv2.dilate(mask2, None, iterations=1)
        mask3 = cv2.inRange(cut_hsv3, hsv_Lower, hsv_Upper)
        mask3 = cv2.erode(mask3, None, iterations=1)
        mask3 = cv2.dilate(mask3, None, iterations=1)
        # mask = cv2.GaussianBlur(mask, (3, 3), 2)  # softly

        cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts3 = cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        center = None

        if len(cnts1) > 0:
            c = max(cnts1, key=cv2.contourArea)
            ((X, Y), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(cut_frame, (int(X), int(Y)), int(radius), (0, 0, 255), 2)
            Area = cv2.contourArea(c) / min_area[now_color]
            if Area > 255:
                Area = 255

            if Area > min_area[now_color]:
                x4, y4, w4, h4 = cv2.boundingRect(c)
                cv2.rectangle(cut_frame, (x4, y4), (x4 + w4, y4 + h4), (0, 255, 0), 2)
                # ----------------------------------------
                rows, cols = cut_frame.shape[:2]
                [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
                # print("rows = " + str(rows) + ", cols= " + str(cols) + ", vx= " + str(vx) + ", vy= " + str(vy) + ", x=" + str(x) + ", y= " + str(y))

                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)

                try:
                    cv2.line(cut_frame, (cols - 1, righty), (0, lefty), (0, 0, 255), 2)
                except:
                    print("cv2.line error~ " + str(righty) + ", " + str(lefty))
                    pass
                point1 = (cols - 1, righty)
                point2 = (0, lefty)

                Angle = 100 + int(GetAngleTwoPoints(point2, point1))

                # ----------DO---------------------------------
                print(Angle)
                # ----------------------------------------

                X_Size = int((255.0 / W_View_size) * w4)
                Y_Size = int((255.0 / H_View_size) * h4)
                X_255_point = int((255.0 / W_View_size) * X)
                Y_255_point = int((255.0 / H_View_size) * Y)

                if mask.color.BLUE:
                    TX_data(serial_port, )
                    TX_data(serial_port, )
                    break
        else:

        if len(cnts2) > 0:
            c = max(cnts2, key=cv2.contourArea)
            ((X, Y), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(cut_frame, (int(X), int(Y)), int(radius), (0, 0, 255), 2)
            Area = cv2.contourArea(c) / min_area[now_color]
            if Area > 255:
                Area = 255

            if Area > min_area[now_color]:
                x4, y4, w4, h4 = cv2.boundingRect(c)
                cv2.rectangle(cut_frame, (x4, y4), (x4 + w4, y4 + h4), (0, 255, 0), 2)
                # ----------------------------------------
                rows, cols = cut_frame.shape[:2]
                [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
                # print("rows = " + str(rows) + ", cols= " + str(cols) + ", vx= " + str(vx) + ", vy= " + str(vy) + ", x=" + str(x) + ", y= " + str(y))

                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)

                try:
                    cv2.line(cut_frame, (cols - 1, righty), (0, lefty), (0, 0, 255), 2)
                except:
                    print("cv2.line error~ " + str(righty) + ", " + str(lefty))
                    pass
                point1 = (cols - 1, righty)
                point2 = (0, lefty)

                Angle = 100 + int(GetAngleTwoPoints(point2, point1))

                # ----------DO---------------------------------
                print(Angle)
                # ----------------------------------------

                X_Size = int((255.0 / W_View_size) * w4)
                Y_Size = int((255.0 / H_View_size) * h4)
                X_255_point = int((255.0 / W_View_size) * X)
                Y_255_point = int((255.0 / H_View_size) * Y)

                if mask.color.YELLOW:
                    if 60 < Angle < 80:
                        TX_data(serial_port, 6)
                    elif Angle > 110:
                        TX_data(serial_port, 4)
                    elif 0 < Angle < 60:
                        TX_data(serial_port, 미정_왼쪽턴45)
                    else:
                        TX_data(serial_port, 2)
                        break

                 if mask.color.GREEN:
                    TX_data(serial_port, 미정_왼쪽턴45)
                    TX_data(serial_port, 3)
                    break

                if mask.color.:
                    TX_data(serial_port, 누워서가기45)
                    break
            else:

            if len(cnts3) > 0:
                c = max(cnts, key=cv2.contourArea)
                ((X, Y), radius) = cv2.minEnclosingCircle(c)
                cv2.circle(cut_frame, (int(X), int(Y)), int(radius), (0, 0, 255), 2)
                Area = cv2.contourArea(c) / min_area[now_color]
                if Area > 255:
                    Area = 255

                if Area > min_area[now_color]:
                    x4, y4, w4, h4 = cv2.boundingRect(c)
                    cv2.rectangle(cut_frame, (x4, y4), (x4 + w4, y4 + h4), (0, 255, 0), 2)
                    # ----------------------------------------
                    rows, cols = cut_frame.shape[:2]
                    [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
                    # print("rows = " + str(rows) + ", cols= " + str(cols) + ", vx= " + str(vx) + ", vy= " + str(vy) + ", x=" + str(x) + ", y= " + str(y))

                    lefty = int((-x * vy / vx) + y)
                    righty = int(((cols - x) * vy / vx) + y)

                    try:
                        cv2.line(cut_frame, (cols - 1, righty), (0, lefty), (0, 0, 255), 2)
                    except:
                        print("cv2.line error~ " + str(righty) + ", " + str(lefty))
                        pass
                    point1 = (cols - 1, righty)
                    point2 = (0, lefty)

                    Angle = 100 + int(GetAngleTwoPoints(point2, point1))

                    # ----------DO---------------------------------
                    print(Angle)
                    # ----------------------------------------

                    X_Size = int((255.0 / W_View_size) * w4)
                    Y_Size = int((255.0 / H_View_size) * h4)
                    X_255_point = int((255.0 / W_View_size) * X)
                    Y_255_point = int((255.0 / H_View_size) * Y)

                    if mask.color.BLUE:
                        TX_data(serial_port, )
                        TX_data(serial_port, )
                        break
            else:

            x = 0
            y = 0
            X_255_point = 0
            Y_255_point = 0
            X_Size = 0
            Y_Size = 0
            Area = 0
            Angle = 0

        # --------------------------------------

        Read_RX = RX_data(serial_port)
        if Read_RX != 0:
            print("Read_RX = " + str(Read_RX))

        # TX_data(serial_port,255)

        # --------------------------------------

        Frame_time = (clock() - old_time) * 1000.
        old_time = clock()

        if View_select == 0:  # Fast operation
            # print(" " + str(W_View_size) + " x " + str(H_View_size) + " =  %.1f ms  Angle: %.2f" % (Frame_time , Angle))
            # temp = Read_RX
            pass

        elif View_select == 1:  # Debug
            draw_str2(frame, (3, 15),
                      'X: %.1d, Y: %.1d, Area: %.1d, Angle: %.2f ' % (X_255_point, Y_255_point, Area, Angle))
            draw_str2(frame, (3, H_View_size - 5), 'View: %.1d x %.1d Time: %.1f ms  Space: Fast <=> Video and Mask.'
                      % (W_View_size, H_View_size, Frame_time))

            # ------------------------------------------
            mx2 = mx
            my2 = my
            pixel = hsv[my2, mx2]
            set_H = pixel[0]
            set_S = pixel[1]
            set_V = pixel[2]
            pixel2 = frame[my2, mx2]

            # frame[2*height//3:,:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            if my2 < (H_View_size / 2):
                if mx2 < 50:
                    x_p = -30
                elif mx2 > (W_View_size - 50):
                    x_p = 60
                else:
                    x_p = 30

                draw_str2(frame, (mx2 - x_p, my2 + 15), '-HSV-')
                draw_str2(frame, (mx2 - x_p, my2 + 30), '%.1d' % (pixel[0]))
                draw_str2(frame, (mx2 - x_p, my2 + 45), '%.1d' % (pixel[1]))
                draw_str2(frame, (mx2 - x_p, my2 + 60), '%.1d' % (pixel[2]))
            else:
                x_p = 30
                draw_str2(frame, (mx2 - x_p, my2 - 60), '-HSV-')
                draw_str2(frame, (mx2 - x_p, my2 - 45), '%.1d' % (pixel[0]))
                draw_str2(frame, (mx2 - x_p, my2 - 30), '%.1d' % (pixel[1]))
                draw_str2(frame, (mx2 - x_p, my2 - 15), '%.1d' % (pixel[2]))

            cv2.imshow('Kongdols(frame) - Video', frame)
            cv2.imshow('Kongdols(mask) - Mask', mask)

            # ----------------------------------------------

        key = 0xFF & cv2.waitKey(1)

        if key == 27:  # ESC  Key
            break
        elif key == ord(' '):  # spacebar Key
            if View_select == 0:
                View_select = 1

            else:
                View_select = 0
    cv2.destroyAllWindows()
    # cleanup the camera and close any open windows
    if serial_use != 0:
        serial_port.close()
        camera.release()
        cv2.destroyAllWindows()

