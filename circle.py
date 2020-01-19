class Circle (object):
    def __init__ (self, x, y, ctype):
        if x:
            self.x = x
        else:
            self.x = 0

        if y:
            self.y = y
        else:
            self.y = 0

        #print("adding " + ctype + " at: ( " + str(self.x) + " | " + str(self.y) + " )")