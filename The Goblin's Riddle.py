# First import the robot object
import json
import random

from mistyPy.Robot import Robot
import mediapipe as mp
import cv2
import time
import base64
from PIL import Image
import io


class HandDetector:
    """
    使用mediapipe库查找手。导出地标像素格式。添加了额外的功能。
    如查找方式，许多手指向上或两个手指之间的距离。而且提供找到的手的边界框信息。
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: 在静态模式下，对每个图像进行检测
        :param maxHands: 要检测的最大手数
        :param detectionCon: 最小检测置信度
        :param minTrackCon: 最小跟踪置信度
        """
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = False
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        # 初始化手部识别模型
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils  # 初始化绘图器
        self.tipIds = [4, 8, 12, 16, 20]  # 指尖列表
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True):
        """
        从图像(BRG)中找到手部。
        :param img: 用于查找手的图像。
        :param draw: 在图像上绘制输出的标志。
        :return: 带或不带图形的图像
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将传入的图像由BGR模式转标准的Opencv模式——RGB模式，
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """
        查找单手的地标并将其放入列表中像素格式。还可以返回手部周围的边界框。
        :param img: 要查找的主图像
        :param handNo: 如果检测到多只手，则为手部id
        :param draw: 在图像上绘制输出的标志。(默认绘制矩形框)
        :return: 像素格式的手部关节位置列表；手部边界框
        """

        xList = []
        yList = []
        bbox = []
        bboxInfo = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                px, py = int(lm.x * w), int(lm.y * h)
                xList.append(px)
                yList.append(py)
                self.lmList.append([px, py])
                if draw:
                    cv2.circle(img, (px, py), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            cx, cy = bbox[0] + (bbox[2] // 2), \
                     bbox[1] + (bbox[3] // 2)
            bboxInfo = {"id": id, "bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                              (0, 255, 0), 2)

        return self.lmList, bboxInfo

    def fingersUp(self):
        """
        查找列表中打开并返回的手指数。会分别考虑左手和右手
        ：return：竖起手指的列表
        """
        if self.results.multi_hand_landmarks:
            myHandType = self.handType()
            fingers = []
            # Thumb
            if myHandType == "Right":
                if self.lmList[self.tipIds[0]][0] > self.lmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if self.lmList[self.tipIds[0]][0] < self.lmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # 4 Fingers
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][1] < self.lmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def handType(self):
        """
        检查传入的手部是左还是右
        ：return: "Right" 或 "Left"
        """
        if self.results.multi_hand_landmarks:
            if self.lmList[17][0] < self.lmList[5][0]:
                return "Right"
            else:
                return "Left"

def predict_photo():
    print("成功进入函数")
    img = cv2.imread("./1.jpg")
    num = 1
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('frame', gray)
    detector = HandDetector()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if lmList:
        x_1, y_1 = bbox["bbox"][0], bbox["bbox"][1]
        x1, x2, x3, x4, x5 = detector.fingersUp()

        if (x2 == 1 and x3 == 1) and (x4 == 0 and x5 == 0 and x1 == 0):
            cv2.putText(img, "2_TWO", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 3)
            num = 2
        elif (x2 == 1 and x3 == 1 and x4 == 1) and (x1 == 0 and x5 == 0):
            cv2.putText(img, "3_THREE", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 3)
            num = 3
        elif (x2 == 1 and x3 == 1 and x4 == 1 and x5 == 1) and (x1 == 0):
            cv2.putText(img, "4_FOUR", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 3)
            num = 4
        elif x1 == 1 and x2 == 1 and x3 == 1 and x4 == 1 and x5 == 1:
            cv2.putText(img, "5_FIVE", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 3)
            num = 5
        elif x2 == 1 and (x1 == 0, x3 == 0, x4 == 0, x5 == 0):
            cv2.putText(img, "1_ONE", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 3)
            num = 1
        elif x1 and (x2 == 0, x3 == 0, x4 == 0, x5 == 0):
            cv2.putText(img, "GOOD!", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 3)
            num = 6

    print("成功预测")
    cv2.imshow("camera", img)
    cv2.waitKey(5000)
    return num

def take_photo():
    return_json = misty.take_picture(base64=True,displayOnScreen=True,width=640,height=480)
    base64_json = return_json.json()
    base64_pic = base64_json['result']['base64']

    img = base64_to_image(base64_pic)
    save_image(img,"./1.jpg")

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return image_data

def save_image(image_data, file_path):
    image = Image.open(io.BytesIO(image_data))
    image.save(file_path)

def trans_num(num):
    if (num==0):
        return "zero"
    if (num==1):
        return "one"
    if (num==2):
        return "two"
    if (num==3):
        return "three"
    if (num==4):
        return "four"
    if (num==5):
        return "five"
    if (num==6):
        return "six"

if __name__ == "__main__":

# 配置meidapipe
    hand_drawing_utils = mp.solutions.drawing_utils  # 绘图工具
    mp_hands = mp.solutions.hands  # 手部识别api
    my_hands = mp_hands.Hands()  # 获取手部识别类
    # print(predict_photo())
    # predict_photo()
# 配置
    ip_address = '192.168.76.208'
    # Create an instance of a robot
    misty = Robot('192.168.76.208')

    # hh = misty.start_recording_video(width=640,height=480)
    # time.sleep(3)
    # print(hh)
    # hh = misty.stop_recording_video()
    # print(hh)
    # hh = misty.get_video_recording()
    # time.sleep(1)
    # print(hh)
    # hh = misty.display_video()
    # time.sleep(3)
    # print(hh)

# step1：
    # 先自我介绍
    misty.start_action("Hi")
    misty.drive_time(10,0,2000)
    misty.speak(text="Welcome to the world of adventure. i am misty and I am your guide through this realm, and together, "
                     "we shall embark on a journey filled with mysteries, choices, and unforgettable stories.",speechRate=0.9)
    time.sleep(12)
    misty.speak(text="In the realm of D&D, you are not confined to the limits of the everyday; you can be anyone you wish. Three distinct paths await you and you can pose your finger to choose from knight, rogue or wizard:",speechRate=0.9)
    misty.start_action("head-nod-slow")
    misty.drive_time(-10,0,2000)
    time.sleep(10)
#
    time.sleep(5)
    take_photo()
    results = predict_photo()
    print(results)
    time.sleep(5)

    # 1,2,3 指123种手势
    if results == 1:
        misty.speak(text="The valiant knight, whose courage and honor shine like a beacon.",speechRate=0.9)
        misty.start_action("head-nod-slow")
        misty.start_action("walk-fast")
        time.sleep(5)
    elif results == 2:
        misty.speak(text="The cunning rogue, a master of stealth and deception.",speechRate=0.9)
        misty.start_action("head-nod-slow")
        misty.start_action("admiration")
        time.sleep(5)
    elif results == 3:
        misty.speak(text="The wise wizard, a wielder of arcane power and boundless knowledge.",speechRate=0.9)
        misty.start_action("head-nod-slow")
        misty.start_action("admire")
        time.sleep(5)
    else:
        misty.speak(text="Well I think you would like to be Knight",speechRate=0.9)
        misty.start_action("head-nod-slow")
        misty.start_action("admiration")
        time.sleep(5)

# # step2：
    misty.speak(text="ok, Now our adventure shall begin!",speechRate=0.9)
    misty.start_action("think")
    misty.start_action("default")
    time.sleep(3)
    misty.speak(text="As you venture deeper into the dense forest, the world around you becomes a tapestry of emerald and shadow.",speechRate=0.9)
    misty.start_action("cute")
    misty.start_action("love")
    time.sleep(8)
    misty.speak(text="Before you stands a mischievous goblin, If you answer my question, I'll let you pass. Fail, and you'll have to do a silly dance!",speechRate=0.9)
    misty.start_action("cry-slow")
    misty.start_action("terror")
    time.sleep(8)
# step3：
    misty.speak(text="It smirks and says, 'Are you ready for a challenge, brave adventurer?'",speechRate=0.9)
    misty.start_action("concerned")
    misty.start_action("sleepy")
    time.sleep(8)

    #dice = random.randint(1,6)
    dice = 6
    print(dice)
    if results == 1:
        misty.speak(text="But luck for you, my adventure, you can do a strength check to path the goblin cause you are a knight",speechRate=0.9)
        misty.start_action("joy")
        time.sleep(8)
    elif results == 2:
        misty.speak(text="But luck for you, my adventure, you can do a speed check to path the goblin cause you are a rogue",speechRate=0.9)
        misty.start_action("joy")
        time.sleep(8)
    elif results == 3:
        misty.speak(text="But luck for you, my adventure, you can do a magic check to path the goblin cause you are a wizard",speechRate=0.9)
        misty.start_action("joy")
        time.sleep(8)
    else:
        misty.speak(text="But luck for you, my adventure, you can do a strength check to path the goblin cause you are a knight",speechRate=0.9)
        misty.start_action("joy")
        time.sleep(8)

    #生成色子图片并展示
    if dice > 4:
        str = "You have a " + trans_num(dice)
        misty.speak(text=str)
        misty.speak(text="congratuation,you use your power knock down the goblin. and the path is just behind him, you can continue your journey now!",speechRate=0.9)
        misty.start_action("admire")
        misty.start_action("admiration")
        time.sleep(8)
    else:
        str = "You have a " + trans_num(dice)
        misty.speak(text=str)
        misty.speak(text="The goblin poses a question, a test of your wits and intelligence. ",speechRate=0.9)
        misty.start_action("admire")
        misty.start_action("admiration")
        time.sleep(5)
        # 加讲故事姿势
        time.sleep(5)
        #加随机数生成
        rand_1 = random.randint(0,3)
        rand_2 = random.randint(0,2)
        sum = rand_1 + rand_2
        texts = "what is the result of "+trans_num(rand_1) + "and" + trans_num(rand_2)
        time.sleep(5)
        misty.speak(text=texts)
        misty.start_action("think")
        misty.start_action("default")
        # 加困惑姿势
        time.sleep(10)
        #再识别一次手势得到答案
        take_photo()
        results_3 = predict_photo()

        if results_3 ==sum:
            misty.speak(text="'Impressive!' it exclaims, and with a theatrical bow, it steps aside, clearing your path.",speechRate=0.9)
            misty.start_action("angry")
            misty.start_action("hi")
            time.sleep(8)
        else:
            misty.speak(text="Unfortunately, you have to dance for me now!",speechRate=0.9)
            misty.start_action("listen")
            misty.start_action("love")
            time.sleep(8)
    misty.speak(text="You continue your journey with a sense of accomplishment and curiosity, eager to uncover the adventures that await you beyond the next turn in the path.",speechRate=0.9)
    # 加恭喜欢迎姿势
    time.sleep(5)

# step4：
    misty.speak(text="As the laughter of the goblin fades into the distance, you find yourself standing at the precipice of a world filled with wonder and adventure. "
                     "This, my friend, is the essence of Dungeons & Dragons.",speechRate=0.9)
    misty.start_action("mad")
    time.sleep(5)
    #加讲解姿势
    time.sleep(5)
    misty.speak(text="In our brief encounter, you've glimpsed the core elements of the game: storytelling, character creation, decision-making, and the thrill of facing challenges."
                     " You've taken your first steps into a world where your choices shape the narrative, and your imagination knows no bounds.",speechRate=0.9)
    time.sleep(5)