#  +------------------------+
#  |        Snake-AI        |
#  |                        |
#  |    von Konrad Weiß     |
#  +------------------------+
from settings import Settings           # settings.py with initial settings
from circle import Circle               # circle.py with class for all displayed objects
from ai import MultiLayerPerceptron

# import used kivy parts
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle,Canvas,Ellipse,Color
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.window import Window

# random, numpy, sys and time
from random import randrange
import numpy as np
import time
import sys
import json
import socket
import sqlite3

HOST = '192.168.1.84'  # Standard loopback interface address (localhost)
PORT = 31415        # Port to listen on (non-privileged ports are > 1023)

# grid for game content
class DrawingArea(GridLayout):

    def update (self, dt):
        app.visAI.draw()
        text = "highscore: " + str(app.game.settings.highscore) + "\nlenght: " + str(app.game.score) + "\nticks: " + str(app.game.ticks) + "\nspeed: " + str(app.speed) + "\nAIcontrolling: " + str(app.game.settings.aiControlled) + "\nrandom: " + str(app.game.settings.random) + "\ncircling: " + str(app.game.settings.circling)
        self.labelInfo.text = text
        if app.game.settings.running:
            app.game.updateScreen()
            self.food.pos = (55 + (app.game.food.x * 30), 55 + (app.game.food.y * 30))
            i = 0
            while i < len(app.game.snake):
                self.snake[i].pos = (55 + (app.game.snake[i].x * 30), 55 + (app.game.snake[i].y * 30))
                i += 1

            app.game.ticks += 1
            print("x:" + str(app.game.snake[0].x) + " | y:" + str(app.game.snake[0].y))
            self.aiSensorDown.text = "wall: " + str(app.game.aiSensors[0][0]) + "\nbody: " + str(app.game.aiSensors[0][1]) + "\nfood: " + str(app.game.aiSensors[0][2])
            self.aiSensorRight.text = "wall: " + str(app.game.aiSensors[1][0]) + "\nbody: " + str(app.game.aiSensors[1][1]) + "\nfood: " + str(app.game.aiSensors[1][2])
            self.aiSensorUp.text = "wall: " + str(app.game.aiSensors[2][0]) + "\nbody: " + str(app.game.aiSensors[2][1]) + "\nfood: " + str(app.game.aiSensors[2][2])
            self.aiSensorLeft.text = "wall: " + str(app.game.aiSensors[3][0]) + "\nbody: " + str(app.game.aiSensors[3][1]) + "\nfood: " + str(app.game.aiSensors[3][2])
            

    def __init__ (self, **kwargs):
        super(DrawingArea, self).__init__(**kwargs)

        with self.canvas:
            # Box
            Color(0.3,0.3,0.3,1)
            self.box = Rectangle(pos=(50, 50), size=(app.game.settings.width * 30, app.game.settings.height * 30))
            Color(1,0,0,1)
            self.food = Ellipse(pos=(0, 0), size=(20, 20))
            Color(1,1,1,1)
            self.snake = []
            self.head = Ellipse(pos=(0, 0), size=(20, 20))
            self.snake.append(self.head)
            self.labelInfo = Label(pos=(app.game.settings.width * 30 + 50, 50))

    def addTile (self):
        with self.canvas:
            Color(1,1,1,1)
            self.tile = Ellipse(pos=(0, 0), size=(20, 20))
            self.snake.append(self.tile)

    def startGame(self):
        self.canvas.clear()
        with self.canvas:
            Color(0.3,0.3,0.3,1)
            self.box = Rectangle(pos=(50, 50), size=(app.game.settings.width * 30, app.game.settings.height * 30))
            Color(0,1,0,1)
            self.food = Ellipse(pos=(0, 0), size=(20, 20))
            Color(1,1,1,1)
            self.snake = []
            self.head = Ellipse(pos=(0, 0), size=(20, 20))
            self.snake.append(self.head)
            self.labelInfo = Label(pos=(app.game.settings.width * 30 + 80, 30))
            self.aiSensorHeaderDown = Label(pos=(app.game.settings.width * 30 + 470, 275), bold=True, text="DOWN:")
            self.aiSensorDown = Label(pos=(app.game.settings.width * 30 + 470, 240))
            self.aiSensorHeaderRight = Label(pos=(app.game.settings.width * 30 + 470, 195), bold=True, text="RIGHT:")
            self.aiSensorRight = Label(pos=(app.game.settings.width * 30 + 470, 160))
            self.aiSensorHeaderUp = Label(pos=(app.game.settings.width * 30 + 470, 115), bold=True, text="UP:")
            self.aiSensorUp = Label(pos=(app.game.settings.width * 30 + 470, 80))
            self.aiSensorHeaderLeft = Label(pos=(app.game.settings.width * 30 + 470, 35), bold=True, text="LEFT:")
            self.aiSensorLeft = Label(pos=(app.game.settings.width * 30 + 470, 0))

# widget for network visualisation
class visAI (Widget):
    def __init__ (self, **kwargs):
        super(visAI, self).__init__(**kwargs)
        self.draw()

    def draw (self):
        with self.canvas:
            # border and box
            Color(1,1,1,1)
            self.border = Rectangle(pos=(app.game.settings.width * 30 + 60, 340), size=(500, 430))
            Color(0,0,0,1)
            self.box = Rectangle(pos=(app.game.settings.width * 30 + 62, 342), size=(496, 426))

            if hasattr(app, 'ai'):
                # draw input neurons and save in array
                self.inputNeurons = []
                for i in range(13):
                    Color(1,1,1,(app.ai.network[0][i] / 24))
                    if i == 12:
                        # if bias color red
                        Color(1,0.1,0.1,1)
                    neuron = Ellipse(pos=(app.game.settings.width * 30 + 62 + 20, 342 + 40 + i * 27), size=(23, 23))
                    self.inputNeurons.append(neuron)
                
                # draw hidden neurons and save in array
                Color(1,1,1,1)
                self.hiddenNeurons = []
                for i in range(17):
                    if i == 16:
                        # if bias color red
                        Color(1,0.1,0.1,1)
                    neuron = Ellipse(pos=(app.game.settings.width * 30 + 62 + 240, 342 + 5 + i * 24), size=(23, 23))
                    self.hiddenNeurons.append(neuron)

                # draw output neurons and save in array
                Color(1,1,1,1)
                self.outputNeurons = []
                for i in range(4):
                    neuron = Ellipse(pos=(app.game.settings.width * 30 + 62 + 460, 342 + 130 + i * 40), size=(23, 23))
                    self.outputNeurons.append(neuron)


class Controller (Widget):
    def __init__ (self, **kwargs):
        super(Controller, self).__init__(**kwargs)

        # setup keyboard controlle
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        self.btn_up = Button(text='up', on_press=self.changeDirectionUp, pos=(app.game.settings.width * 30 + 270, 220))
        self.btn_left = Button(text='left', on_press=self.changeDirectionLeft, pos=(app.game.settings.width * 30 + 170, 120))
        self.btn_down = Button(text='down', on_press=self.changeDirectionDown, pos=(app.game.settings.width * 30 + 270, 20))
        self.btn_right = Button(text='right', on_press=self.changeDirectionRight, pos=(app.game.settings.width * 30 + 370, 120))
        self.btn_break = Button(text='break', on_press=self.breakGame, pos=(app.game.settings.width * 30 + 270, 120))
        self.btn_ai = Button(text='AIcontrolling', on_press=self.aiControlling, pos=(app.game.settings.width * 30 + 60, 150), size=(100,30))
        self.btn_rnd = Button(text='random', on_press=self.random, pos=(app.game.settings.width * 30 + 60, 185), size=(100,30))
        self.btn_circle = Button(text='circle', on_press=self.circling, pos=(app.game.settings.width * 30 + 60, 220), size=(100,30))

        self.add_widget(self.btn_up)
        self.add_widget(self.btn_left)
        self.add_widget(self.btn_down)
        self.add_widget(self.btn_right)
        self.add_widget(self.btn_break)
        self.add_widget(self.btn_ai)
        self.add_widget(self.btn_rnd)
        self.add_widget(self.btn_circle)

    
    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'left':
            self.changeDirection(direction = 3)
        elif keycode[1] == 'right':
            self.changeDirection(direction = 1)
        elif keycode[1] == 'up':
            self.changeDirection(direction = 2)
        elif keycode[1] == 'down':
            self.changeDirection(direction = 0)
        return True

    def changeDirectionUp (self, d):
        self.changeDirection(direction = 2)
    def changeDirectionLeft (self, d):
        self.changeDirection(direction = 3)
    def changeDirectionDown (self, d):
        self.changeDirection(direction = 0)
    def changeDirectionRight (self, d):
        self.changeDirection(direction = 1)
    def breakGame (self, d):
        if app.game.settings.running:
            app.game.settings.running = False
        else:
            app.game.settings.running = True
        
    def random (self, d):
        # activate / deactivate random function when choosen
        if app.game.settings.random:
            app.game.settings.random = False
        else:
            app.game.settings.random = True

            # on activation deactivate all others
            app.game.settings.circling = False
            app.game.settings.aiControlled = False

            # faster speed for random
            Clock.unschedule(app.clock)
            app.speed = 20
            app.clock = Clock.schedule_interval(app.drawingArea.update, (app.speed * 0.0025))
        
    def circling (self, d):
        # activate / deactivate circling when choosen
        if app.game.settings.circling:
            app.game.settings.circling = False
        else:
            app.game.settings.circling = True

            # on activation deactivate all others
            app.game.settings.random = False
            app.game.settings.aiControlled = False

    def aiControlling (self, d):
        # activate / deactivate ai controle when choosen
        if app.game.settings.aiControlled:
            app.game.settings.aiControlled = False
        else:
            app.game.settings.aiControlled = True

            # on activation deactivate all others
            app.game.settings.circling = False
            app.game.settings.random = False

    def changeDirection (self, direction):
        # if player is playing save direction in database
        if not (app.game.settings.aiControlled or app.game.settings.circling or app.game.settings.random):
            app.game.newDirection = direction
            if app.game.settings.saveSteps:
                print("save: " + str(direction))
                dataString = "(" + str(app.game.aiSensors[0][0]) + "," + str(app.game.aiSensors[0][1]) + "," + str(app.game.aiSensors[0][2]) + "," + str(app.game.aiSensors[1][0]) + "," + str(app.game.aiSensors[1][1]) + "," + str(app.game.aiSensors[1][2]) + "," + str(app.game.aiSensors[2][0]) + "," + str(app.game.aiSensors[2][1]) + "," + str(app.game.aiSensors[2][2]) + "," + str(app.game.aiSensors[3][0]) + "," + str(app.game.aiSensors[3][1]) + "," + str(app.game.aiSensors[3][2]) + "," + str(direction) + ")"
                sql_command = "INSERT INTO testData (aiSensor_0_0, aiSensor_0_1, aiSensor_0_2, aiSensor_1_0, aiSensor_1_1, aiSensor_1_2, aiSensor_2_0, aiSensor_2_1, aiSensor_2_2, aiSensor_3_0, aiSensor_3_1, aiSensor_3_2, direction) VALUES " + dataString + ";"

                # add direction changes to learndata in db
                app.cursor.execute(sql_command)
                app.db.commit()

# main application
class SnakeApp (App):
    def build (self):
        self.game = SnakeGame()             # initialize game stuff
        self.drawingArea = DrawingArea()    # create drawing area
        self.controller = Controller()      # create controller
        self.visAI = visAI()

        # test weights
        W_IH = np.matrix([[-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22],[-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22],[-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22],[-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]])
        W_HO = np.matrix([[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22],[-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,-0.22,1.5,1.5]])
        weights = []
        weights.append(W_IH)
        weights.append(W_HO)

        # initialise MLP
        self.ai = MultiLayerPerceptron(12, 16, 4, weights)

        self.speed = self.game.settings.speed

        # clock executing gametick for speed from settings
        self.clock = Clock.schedule_interval(self.drawingArea.update, (self.speed * 0.0025))

        # add drawing area, controller and grid layout to app
        root = GridLayout(cols=3)
        root.add_widget(self.drawingArea)
        root.add_widget(self.controller)
        root.add_widget(self.visAI)

        if app.game.settings.saveSteps:
            self.db = sqlite3.connect("file:trainingData.db?mode=rwc", uri=True)
            self.cursor = self.db.cursor()

        return root

# main game class
class SnakeGame (object):
    def __init__ (self):
        self.settings = Settings()              # create setting
        self.food = Circle(0, 0, "food-dummy")  # init food (real position later)
        self.highscore = 0
        self.score = 3
        self.ticks = 0
        self.aiSensors = []                 # array with all ai needed inputs
        self.aiSensors.append(np.zeros(3))  # down      -> wall, body, food
        self.aiSensors.append(np.zeros(3))  # right     -> wall, body, food
        self.aiSensors.append(np.zeros(3))  # top       -> wall, body, food
        self.aiSensors.append(np.zeros(3))  # left      -> wall, body, food
        
        if self.settings.visOnPi:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((HOST,PORT))
            s.listen()
            self.conn, addr = s.accept()
            print('Connected by', addr)
        else:
            self.conn = None
            addr = None
            s = None

    # after death on start
    def startGame (self):
        self.snake = []             # clear snake
        self.newDirection = 0
        app.drawingArea.startGame() # start drawing area
        head = Circle(round((self.settings.width) / 2), round((self.settings.height) / 2), "head")
        self.snake.append(head)     # add head to snake
        self.settings.gaming = True # set game to running
        self.ticks = 0

        self.generateFood()         # create food
        self.eat()                  # eat 2x for start lenght 3
        self.eat()

    def generateFood (self):
        rndx = randrange(0, self.settings.width)    # random x position
        rndy = randrange(0, self.settings.height)   # random y position

        self.food = Circle(rndx, rndy, "food")      # create food as circle

    def eat (self):
        # new tile
        lenght = len(self.snake)
        tlx = self.snake[lenght - 1].x
        tly = self.snake[lenght - 1].y
        body = Circle(tlx, tly, "tile-" + str(lenght))
        self.snake.append(body)     # add tile to snake
        app.drawingArea.addTile()   # create new circle in grafik
        self.settings.score += self.settings.points
        self.generateFood()         # create new food spot
        self.score = lenght + 1     # set score to lenght
        Clock.unschedule(app.clock)
        app.speed += -1
        app.clock = Clock.schedule_interval(app.drawingArea.update, (app.speed * 0.0025))

    def die (self):
        # update highscore if score is higher
        if self.settings.highscore < self.score:
            self.settings.highscore = self.score
        # print values on death
        print("lenght:    " + str(self.score))
        print("highscore: " + str(self.settings.highscore))
        print("========================")

        # reset clock/speed
        if app.game.settings.random:
            Clock.unschedule(app.clock)
            app.speed = 20
            app.clock = Clock.schedule_interval(app.drawingArea.update, (app.speed * 0.0025))
        else:
            Clock.unschedule(app.clock)
            app.speed = app.game.settings.speed
            app.clock = Clock.schedule_interval(app.drawingArea.update, (app.speed * 0.0025))

        self.settings.gaming = False    # stop game

    def movePlayer (self):
        if self.settings.random:
            self.newDirection = randrange(0, 4)
            if self.aiSensors[0][2]:
                self.newDirection = 0
            if self.aiSensors[1][2]:
                self.newDirection = 1
            if self.aiSensors[2][2]:
                self.newDirection = 2
            if self.aiSensors[3][2]:
                self.newDirection = 3

            if self.aiSensors[0][0] < 1:
                self.newDirection = 1
            if self.aiSensors[3][0] < 1:
                self.newDirection = 0
            if self.aiSensors[1][0] > self.settings.width -1:
                self.newDirection = 2
            if self.aiSensors[2][0] > self.settings.height -1:
                self.newDirection = 3

        # AI CONTROLLING including AI predict
        if self.settings.aiControlled:
            print(self.aiSensors)

            # ai sensor values to input array
            x = []
            x.append(1)
            x.append(self.aiSensors[0][0])
            x.append(self.aiSensors[0][1])
            x.append(self.aiSensors[0][2])
            x.append(self.aiSensors[1][0])
            x.append(self.aiSensors[1][1])
            x.append(self.aiSensors[1][2])
            x.append(self.aiSensors[2][0])
            x.append(self.aiSensors[2][1])
            x.append(self.aiSensors[2][2])
            x.append(self.aiSensors[3][0])
            x.append(self.aiSensors[3][1])
            x.append(self.aiSensors[3][2])

            # calculate output of network
            output = app.ai.predict(x)
            print(output)

            # new direction from network ouput
            self.newDirection = np.argmax(output)
            print(self.newDirection)

        else:
            if self.settings.circling:
                self.newDirection = 3
                if  self.snake[0].x == 0:
                    if self.settings.direction == 3:
                        self.newDirection = 0
                    else:
                        if self.settings.direction == 0:
                            self.newDirection = 1
                else:
                    if self.snake[0].x == self.settings.width - 2:
                        if self.settings.direction == 1:
                            self.newDirection = 0
                        else:
                            if self.settings.direction == 0:
                                self.newDirection = 3
                if self.snake[0].y == 0:
                    self.newDirection = 1
                
                if self.snake[0].x == self.settings.width - 1:
                    if self.snake[0].y == self.settings.height - 1:
                        self.newDirection = 3
                    else:
                        self.newDirection = 2

        if self.settings.direction < 2:
            if self.settings.direction + 2 != self.newDirection:
                self.settings.direction = self.newDirection
        else:
            if self.settings.direction > 1:
                if self.settings.direction - 2 != self.newDirection:
                    self.settings.direction = self.newDirection

        self.aiSensors[0][1:] = 0
        self.aiSensors[1][1:] = 0
        self.aiSensors[2][1:] = 0
        self.aiSensors[3][1:] = 0

        i = len(self.snake) - 1
        while i >= 0:
            if i == 0:
                if self.settings.direction == 0:
                    self.snake[i].y += (-1)
                if self.settings.direction == 1:
                    self.snake[i].x += 1
                if self.settings.direction == 2:
                    self.snake[i].y += 1
                if self.settings.direction == 3:
                    self.snake[i].x += (-1)

                if self.snake[i].x < 0 or self.snake[i].y < 0 or self.snake[i].x >= self.settings.width or self.snake[i].y >= self.settings.height:
                    self.die()
                
                j = 0
                while j < len(self.snake):
                    if j > 0:
                        if self.snake[i].x == self.snake[j].x and self.snake[i].y == self.snake[j].y:
                            self.die()
                    
                    j += 1
                    
                if self.snake[i].x == self.food.x and self.snake[i].y == self.food.y:
                    self.eat()

            else:
                self.snake[i].x = self.snake[i - 1].x
                self.snake[i].y = self.snake[i - 1].y
                
                if self.snake[0].x == self.snake[i].x:
                    if self.snake[0].y > self.snake[i].y:
                        self.aiSensors[0][1] = self.snake[0].y - self.snake[i].y
                    else:
                        if self.snake[0].y < self.snake[i].y:
                            self.aiSensors[2][1] = self.snake[i].y - self.snake[0].y
                
                if self.snake[0].y == self.snake[i].y:
                    if self.snake[0].x > self.snake[i].x:
                        self.aiSensors[3][1] = self.snake[0].x - self.snake[i].x
                    else:
                        if self.snake[0].x < self.snake[i].x:
                            self.aiSensors[1][1] = self.snake[i].x - self.snake[0].x

            i += -1

        self.aiSensors[0][0] = self.snake[0].y + 1
        self.aiSensors[1][0] = self.settings.width - self.snake[0].x
        self.aiSensors[2][0] = self.settings.height - self.snake[0].y
        self.aiSensors[3][0] = self.snake[0].x + 1

        if self.snake[0].x == self.food.x:
            if self.snake[0].y > self.food.y:
                self.aiSensors[0][2] = self.snake[0].y - self.food.y
            else:
                if self.snake[0].y < self.food.y:
                    self.aiSensors[2][2] = self.food.y - self.snake[0].y
        
        if self.snake[0].y == self.food.y:
            if self.snake[0].x > self.food.x:
                self.aiSensors[3][2] = self.snake[0].x - self.food.x
            else:
                if self.snake[0].x < self.food.x:
                    self.aiSensors[1][2] = self.food.x - self.snake[0].x

    def updateScreen (self):
        if self.settings.gaming == False:
            self.startGame()
        else:
            if self.settings.visOnPi and self.conn:
                json_string = "["
                for part in self.snake:
                    json_string += "["
                    json_string += str(part.x)
                    json_string += ","
                    json_string += str(part.y)
                    json_string += "],"
                json_string += "[" + str(self.food.x) + "," + str(self.food.y) + "]"
                json_string += "]"

                data = json_string.encode('UTF-8')
                print(data.decode('UTF-8'))
                self.conn.send(data)

            self.movePlayer()
            
# initiaisation of app and running
app = SnakeApp()
app.run()