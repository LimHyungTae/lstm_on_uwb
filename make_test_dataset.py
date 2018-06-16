import random
import math
class UWB:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def getDistance(self,robot_x, robot_y):
        distance = math.sqrt((self.x -robot_x)**2 + (self.y - robot_y)**2)
        return distance
    def getDistancewNoise(self,robot_x, robot_y):
        distance = math.sqrt((self.x -robot_x)**2 + (self.y - robot_y)**2) + 5*random.random()
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

train_file = open('data_for_test.txt','w')
kobuki = Robot(0, 0)

for i in range(500):

    kobuki.setPose(10,10)

    dist1 = uwb1.getDistancewNoise(kobuki.getPose()[0], kobuki.getPose()[1])
    dist2 = uwb2.getDistancewNoise(kobuki.getPose()[0], kobuki.getPose()[1])
    dist3 = uwb3.getDistancewNoise(kobuki.getPose()[0], kobuki.getPose()[1])
    dist4 = uwb4.getDistancewNoise(kobuki.getPose()[0], kobuki.getPose()[1])
    # dist1 = uwb1.getDistance(kobuki.getPose()[0], kobuki.getPose()[1])
    # dist2 = uwb2.getDistance(kobuki.getPose()[0], kobuki.getPose()[1])
    # dist3 = uwb3.getDistance(kobuki.getPose()[0], kobuki.getPose()[1])
    # dist4 = uwb4.getDistance(kobuki.getPose()[0], kobuki.getPose()[1])
    train_file.write(str(dist1)+' '+str(dist2)+' '+str(dist3)+' '+str(dist4)+' '+ str(kobuki.pose_x)+' ' + str(kobuki.pose_y)+' \n')
    train_file.write(str(dist1)+' '+str(dist2)+' '+str(dist3)+' '+str(dist4)+' '+ str(kobuki.pose_x)+' ' + str(kobuki.pose_y)+' \n')

print(kobuki.getPose())
print (uwb1.getDistance(0,0))
train_file.close()

