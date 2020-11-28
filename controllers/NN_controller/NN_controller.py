from controller import Robot
from datetime import datetime
from random import seed
from random import random
# import numpy as np
from math import exp,log
import json


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network
    

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# test forward propagation
# network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
        # [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
# row = [1, 0, 1,2,3,4,5,8]
# output = forward_propagate(network, row)
# print(output)

    
# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])



def cross_entropy(expected, output):
    for actual,pred in zip(expected, output):
        a = 0
        a -= actual*log(pred) + (1-actual)*log(1-(pred))
        return a



# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']



# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, expected):
    try:
        with open('network.json') as json_file:
            data = json.load(json_file)
        network = data
    except :
        print("no file found")
    outputs = forward_propagate(network, row)
    print(outputs)
    sum_error = cross_entropy(expected, outputs)
    backward_propagate_error(network, expected)
    update_weights(network, row, 1)
    with open('network.json', 'w') as outfile:
        json.dump(network, outfile)

    # print('lrate=%.3f, error=%.3f' % (l_rate, sum_error))
    
    
def predict(network, row):
    try:
        with open('network.json') as json_file:
            data = json.load(json_file)
        network = data
    except :
        print("Cant find file")
    outputs = forward_propagate(network, row)
    with open('network.json', 'w') as outfile:
        json.dump(network, outfile)

    return outputs.index(max(outputs))






# Get reference to the robot.
robot = Robot()


# Get simulation step length.
timeStep = int(robot.getBasicTimeStep())

# Constants of the e-puck motors and distance sensors.
maxMotorVelocity = 4.2
num_left_dist_sensors = 4
num_right_dist_sensors = 4
right_threshold = [75, 75, 75, 75]
left_threshold =  [75, 75, 75, 75]

# Get left and right wheel motors.
leftMotor = robot.getMotor("left wheel motor")
rightMotor = robot.getMotor("right wheel motor")

# Get frontal distance sensors.
dist_left_sensors = [robot.getDistanceSensor('ps' + str(x)) for x in range(num_left_dist_sensors)]  # distance sensors
list(map((lambda s: s.enable(timeStep)), dist_left_sensors))  # Enable all distance sensors

dist_right_sensors = [robot.getDistanceSensor('ps' + str(x)) for x in range(num_right_dist_sensors,8)]  # distance sensors
list(map((lambda t: t.enable(timeStep)), dist_right_sensors))  # Enable all distance sensors

# Disable motor PID control mode.
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

# Set ideal motor velocity.
initialVelocity = 0.7 * maxMotorVelocity

# Set the initial velocity of the left and right wheel motors.
leftMotor.setVelocity(initialVelocity)
rightMotor.setVelocity(initialVelocity)


seed(100)
network = initialize_network(8, 2, 3)

while robot.step(timeStep) != -1:
    y_pred = [0,0,1]
    leftMotor.setVelocity(initialVelocity)
    rightMotor.setVelocity(initialVelocity)
    left_dist_sensor_values = [g.getValue() for g in dist_left_sensors]
    right_dist_sensor_values = [h.getValue() for h in dist_right_sensors]
    
    left_obstacle = [(x > y) for x, y in zip(left_dist_sensor_values, left_threshold)]
    right_obstacle = [(m > n) for m, n in zip(right_dist_sensor_values, right_threshold)]
 
    # if (True in left_obstacle) & (True in right_obstacle):
        # leftMotor.setVelocity(initialVelocity-(0.7*initialVelocity))
        # rightMotor.setVelocity(0*initialVelocity)
     
    
    if True in left_obstacle:
        print(y_pred)
        y_pred[2] = 0
        y_pred[1] = 1
        # print("left True {}".format(datetime.now()))
        # leftMotor.setVelocity(initialVelocity-(0.7*initialVelocity))
        # rightMotor.setVelocity(initialVelocity+(0.5*initialVelocity))
    
    elif True in right_obstacle:
        y_pred[2] = 0
        y_pred[0] = 1
        # print("Right True {}".format(datetime.now()))
        # leftMotor.setVelocity(initialVelocity+(0.7*initialVelocity))
        # rightMotor.setVelocity(initialVelocity-(0.5*initialVelocity))
     
    row = (left_dist_sensor_values+right_dist_sensor_values)
    row = [v/1000 for v in row]
     
    print(predict(network,row))
    
    if predict(network,row) == 0:
        print("Right True {}".format(datetime.now()))
        leftMotor.setVelocity(initialVelocity+(0.7*initialVelocity))
        rightMotor.setVelocity(initialVelocity-(0.5*initialVelocity))
    
    elif predict(network,row) == 1:
        print("left True {}".format(datetime.now()))
        leftMotor.setVelocity(initialVelocity-(0.7*initialVelocity))
        rightMotor.setVelocity(initialVelocity+(0.5*initialVelocity))
    
    train_network(network, row, 0.1, y_pred)
        
        
     