import random
import math
import csv

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
        distance = distance*(1.0 + random.random()* 0.2)
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

train_file = open('train_data_zigzag_3D.csv','w',encoding = 'utf-8', newline ='')

wr = csv.writer(train_file)
kobuki = Robot(0, 0, 0.3)
def moveRobot(x,y,z):
    kobuki.setPose(x, y, z)
    dist1 = uwb1.getDistancewNoise(kobuki)
    dist2 = uwb2.getDistancewNoise(kobuki)
    dist3 = uwb3.getDistancewNoise(kobuki)
    dist4 = uwb4.getDistancewNoise(kobuki)
    return dist1,dist2,dist3,dist4
zigzag= True
if (zigzag):
    pass
    for k in range(1):
        for j in range(2):
            for i in range(500):
                dist_list = moveRobot(0.01, 0.0, 0.0)
                wr.writerow(dist_list + (kobuki.x, kobuki.y, kobuki.z))
            for i in range(125):
                dist_list = moveRobot(0.0, 0.01, 0.0)
                wr.writerow(dist_list + (kobuki.x, kobuki.y, kobuki.z))
            for i in range(500):
                dist_list = moveRobot(-0.01, 0.0, 0.0)
                wr.writerow(dist_list + (kobuki.x, kobuki.y, kobuki.z))
            for i in range(125):
                dist_list = moveRobot(0.0, 0.01, 0.0)
                wr.writerow(dist_list + (kobuki.x, kobuki.y, kobuki.z))


        for i in range(500):

            dist_list = moveRobot(0.01, 0.0, 0.0)
            wr.writerow(dist_list + (kobuki.x, kobuki.y, kobuki.z))

        for j in range(2):
            for i in range(500):
                dist_list = moveRobot(0.0, -0.01, 0.0)
                wr.writerow(dist_list + (kobuki.x, kobuki.y, kobuki.z))

            for i in range(125):
                dist_list = moveRobot(-0.01, 0.0, 0.0)
                wr.writerow(dist_list + (kobuki.x, kobuki.y, kobuki.z))

            for i in range(500):
                dist_list = moveRobot(0.0, 0.01, 0.0)
                wr.writerow(dist_list + (kobuki.x, kobuki.y, kobuki.z))

            for i in range(125):

                dist_list = moveRobot(-0.01, 0.0, 0.0)
                wr.writerow(dist_list + (kobuki.x, kobuki.y, kobuki.z))

        for i in range(500):
            dist_list = moveRobot(0.0,-0.01, 0.0)
            wr.writerow(dist_list + (kobuki.x, kobuki.y, kobuki.z))

else:
    for i in range(100):
        dist_list = moveRobot(0.01, 0.0, 0.0)
        wr.writerow(dist_list + (kobuki.x, kobuki.y, kobuki.z))

    for i in range(100):
        dist_list = moveRobot(0.0, 0.01, 0.0)
        wr.writerow(dist_list + (kobuki.x, kobuki.y, kobuki.z))

    for i in range(100):
        dist_list = moveRobot(-0.01, 0.0, 0.0)
        wr.writerow(dist_list + (kobuki.x, kobuki.y, kobuki.z))

    for i in range(100):
        dist_list = moveRobot(0.0, -0.01, 0.0)
        wr.writerow(dist_list + (kobuki.x, kobuki.y, kobuki.z))
train_file.close()
