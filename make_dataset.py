import random
import math
import csv

ISZIGZAG = True
UNCERTAINTY = 0.1
DIMENSION = '2D'
ROUNDNUM = 20
DELTALENGTH = 0.01
ONESIDELENGTH = 5

class position:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class UWB(position):
    def __init__(self, x, y, z):
        super(UWB, self).__init__(x, y, z)

    def getDistance(self,robot):
        distance = math.sqrt((self.x -robot.x)**2 + (self.y - robot.y)**2 + (self.z -robot.z)**2)
        return distance

    def getDistancewNoise(self, robot):
        distance = self.getDistance(robot)
        distance = distance*(1.0 + random.random()*UNCERTAINTY)

        return distance

class Robot(position):
    def __init__(self, x, y, z):
        super(Robot, self).__init__(x, y, z)

    def getPose(self):
        return (self.x, self.y, self.z)

    def setPose(self,dx,dy,dz):
        self.x += dx
        self.y += dy
        self.z += dz

uwb1 = UWB( 0.9, 0.9, 0)
uwb2 = UWB( 4.5,-0.9, 0)
uwb3 = UWB( 4.5, 4.5, 0)
uwb4 = UWB( 0.9, 2.7, 0)

file_name = 'train_data_' + DIMENSION
if (ISZIGZAG):
    file_name = file_name +'_' + 'zigzag'
file_name = file_name + '.csv'

kobuki = Robot(0, 0, 0.3)
train_file = open(file_name ,'w',encoding = 'utf-8', newline ='')
wr = csv.writer(train_file)

class CSVWriter():
    def __init__(self, wr, kobuki):
        self.dimension = DIMENSION
        self.wr = wr
        self.kobuki = kobuki
        self.iteration_num = int(ONESIDELENGTH/DELTALENGTH)
    def writerow(self, dist_list):
        print (self.kobuki.x, self.kobuki.y)
        if (self.dimension == '2D'):
            self.wr.writerow(dist_list +(self.kobuki.x, self.kobuki.y))

        elif (self.dimension == '3D'):
            self.wr.writerow(dist_list +(self.kobuki.x, self.kobuki.y, self.kobuki.z))

    def moveRobot(self, x,y,z):
        self.kobuki.setPose(x, y, z)
        dist1 = uwb1.getDistancewNoise(kobuki)
        dist2 = uwb2.getDistancewNoise(kobuki)
        dist3 = uwb3.getDistancewNoise(kobuki)
        dist4 = uwb4.getDistancewNoise(kobuki)
        return dist1,dist2,dist3,dist4

    def drawZigzagPath(self, round_number):
        for k in range(round_number):
            for j in range(2):
                for i in range(self.iteration_num):
                    dist_list = self.moveRobot(DELTALENGTH, 0.0, 0.0)
                    self.writerow(dist_list)
                for i in range(int(self.iteration_num/4)):
                    dist_list = self.moveRobot(0.0, DELTALENGTH, 0.0)
                    self.writerow(dist_list)
                for i in range(int(self.iteration_num)):
                    dist_list = self.moveRobot(-DELTALENGTH, 0.0, 0.0)
                    self.writerow(dist_list)
                for i in range(int(self.iteration_num/4)):
                    dist_list = self.moveRobot(0.0, DELTALENGTH, 0.0)
                    self.writerow(dist_list)

            for i in range(self.iteration_num):
                dist_list = self.moveRobot(DELTALENGTH, 0.0, 0.0)
                self.writerow(dist_list)

            for j in range(2):
                for i in range(self.iteration_num):
                    dist_list = self.moveRobot(0.0, -DELTALENGTH, 0.0)
                    self.writerow(dist_list)
                for i in range(int(self.iteration_num / 4)):
                    dist_list = self.moveRobot(-DELTALENGTH, 0.0, 0.0)
                    self.writerow(dist_list)
                for i in range(self.iteration_num):
                    dist_list = self.moveRobot( 0.0,DELTALENGTH, 0.0)
                    self.writerow(dist_list)
                for i in range(int(self.iteration_num / 4)):
                    dist_list = self.moveRobot(-DELTALENGTH, 0.0,  0.0)
                    self.writerow(dist_list)

            for i in range(self.iteration_num):
                dist_list = self.moveRobot(0.0, -DELTALENGTH, 0.0)
                self.writerow(dist_list)

    def drawSquarePath(self, round_number):
        for j in range(round_number):
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(DELTALENGTH, 0.0, 0.0)
                self.writerow(dist_list)
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(0.0, DELTALENGTH, 0.0)
                self.writerow(dist_list)
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(-DELTALENGTH, 0.0, 0.0)
                self.writerow(dist_list)
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(0.0, -DELTALENGTH, 0.0)
                self.writerow(dist_list)


    def drawDiagonalPath(self):
        for i in range(self.iteration_num):
            dist_list = self.moveRobot(DELTALENGTH, DELTALENGTH, 0.0)
            self.writerow(dist_list)

dataWriter = CSVWriter(wr, kobuki)
dataWriter.drawSquarePath(1)

print ("Make "+file_name)


train_file.close()
