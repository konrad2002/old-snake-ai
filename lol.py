
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle,Canvas,Ellipse,Color,Line
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.window import Window

class DrawingArea(GridLayout):
    def __init__ (self, **kwargs):
        super(DrawingArea, self).__init__(**kwargs)
        self.counter = -1
        with self.canvas:
            self.info = Label(pos=(700,350), text=str(self.counter), font_size='100px')

    def update(self, dt):
        if self.counter == 0:
            input(":")
        self.counter += 1
        self.info.text = str(self.counter)
        print(self.counter)


class TutorialApp(App):
    def build(self):
        self.drawingArea = DrawingArea()
        Window.maximize()
        self.clock = Clock.schedule_interval(self.drawingArea.update, 0.000025)
        root = GridLayout(cols=1)
        root.add_widget(self.drawingArea)
        return root

TutorialApp().run()