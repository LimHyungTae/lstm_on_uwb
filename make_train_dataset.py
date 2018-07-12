import random
import math
import csv
class UWB:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def getDistance(self,robot_x, robot_y):
        distance = math.sqrt((self.x -robot_x)**2 + (self.y - robot_y)**2)
        return distance
    def getDistancewNoise(self,robot_x, robot_y):
        distance = math.sqrt((self.x -robot_x)**2 + (self.y - robot_y)**2) + 500.0*random.random()
        return distance
class Robot:
    def __init__(self, x, y):
        self.pose_x = x
        self.pose_y = y

    def getPose(self):
        return (self.pose_x, self.pose_y)
    def setPose(self,dx,dy):
        self.pose_x += dx
        self.pose_y += dy

uwb1 = UWB(-500,-500)
uwb2 = UWB(5500,-500)
uwb3 = UWB(5500,5500)
uwb4 = UWB(-500,5500)

train_file = open('data_train_zigzag.csv','w',encoding = 'utf-8', newline ='')

wr = csv.writer(train_file)
kobuki = Robot(0, 0)
def moveRobot(x,y):
    kobuki.setPose(x, y)
    dist1 = uwb1.getDistancewNoise(kobuki.getPose()[0], kobuki.getPose()[1])
    dist2 = uwb2.getDistancewNoise(kobuki.getPose()[0], kobuki.getPose()[1])
    dist3 = uwb3.getDistancewNoise(kobuki.getPose()[0], kobuki.getPose()[1])
    dist4 = uwb4.getDistancewNoise(kobuki.getPose()[0], kobuki.getPose()[1])
    return dist1,dist2,dist3,dist4
zigzag=True
if (zigzag):
    for k in range(20):
        for j in range(2):
            for i in range(500):
                dist1,dist2,dist3,dist4 = moveRobot(10.0, 0.0)
                wr.writerow([dist1,dist2,dist3,dist4,kobuki.pose_x,kobuki.pose_y])
            for i in range(125):
                dist1,dist2,dist3,dist4 = moveRobot(0.0,10.0)
                wr.writerow([dist1,dist2,dist3,dist4,kobuki.pose_x,kobuki.pose_y])
            for i in range(500):
                dist1,dist2,dist3,dist4 = moveRobot(-10.0, 0.0)
                wr.writerow([dist1,dist2,dist3,dist4,kobuki.pose_x,kobuki.pose_y])
            for i in range(125):
                dist1,dist2,dist3,dist4 = moveRobot(0.0,10.0)
                wr.writerow([dist1,dist2,dist3,dist4,kobuki.pose_x,kobuki.pose_y])
        for i in range(500):
            dist1,dist2,dist3,dist4 = moveRobot(10.0, 0.0)
            wr.writerow([dist1,dist2,dist3,dist4,kobuki.pose_x,kobuki.pose_y])
        for j in range(2):
            for i in range(500):
                dist1,dist2,dist3,dist4 = moveRobot(0.0,-10.0)
                wr.writerow([dist1,dist2,dist3,dist4,kobuki.pose_x,kobuki.pose_y])
            for i in range(125):
                dist1,dist2,dist3,dist4 = moveRobot(-10.0,0.0)
                wr.writerow([dist1,dist2,dist3,dist4,kobuki.pose_x,kobuki.pose_y])
            for i in range(500):
                dist1,dist2,dist3,dist4 = moveRobot(0.0, 10.0)
                wr.writerow([dist1,dist2,dist3,dist4,kobuki.pose_x,kobuki.pose_y])
            for i in range(125):
                dist1,dist2,dist3,dist4 = moveRobot(-10.0,0.0)
                wr.writerow([dist1,dist2,dist3,dist4,kobuki.pose_x,kobuki.pose_y])
        for i in range(500):
            dist1,dist2,dist3,dist4 = moveRobot(0.0,-10.0)
            wr.writerow([dist1,dist2,dist3,dist4,kobuki.pose_x,kobuki.pose_y])

else:
    for i in range(500):
        dist1, dist2, dist3, dist4 = moveRobot(10.0, 0.0)
        wr.writerow([dist1, dist2, dist3, dist4, kobuki.pose_x, kobuki.pose_y])
    for i in range(500):
        dist1, dist2, dist3, dist4 = moveRobot(0.0, 10.0)
        wr.writerow([dist1, dist2, dist3, dist4, kobuki.pose_x, kobuki.pose_y])
    for i in range(500):
        dist1, dist2, dist3, dist4 = moveRobot(-10.0, 0.0)
        wr.writerow([dist1, dist2, dist3, dist4, kobuki.pose_x, kobuki.pose_y])
    for i in range(500):
        dist1, dist2, dist3, dist4 = moveRobot(0.0,-10.0)
        wr.writerow([dist1, dist2, dist3, dist4, kobuki.pose_x, kobuki.pose_y])
train_file.close()


# print(kobuki.getPose())
# print (uwb1.getDistance(0,0))

