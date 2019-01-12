import RPi.GPIO as GPIO
from time import sleep

def set_angle(angle):
    duty = angle/18+2.5
    # turn it on
    GPIO.output(3, True)
    # give it an angle
    pwm.ChangeDutyCycle(duty)
    # give time to get to angle
    sleep(3)
    # turn off the pin
    GPIO.output(3, False)
    pwm.ChangeDutyCycle(0)


# set naming for pins
GPIO.setmode(GPIO.BOARD)
# make pin 3 output PWN
GPIO.setup(3, GPIO.OUT)
# set pin 3 PWN to 5 hz
pwm = GPIO.PWM(3, 50)
# start with zero duty cycle: dont set angle at start
pwm.start(0)

print("dam")
while True:
    set_angle(0)
    set_angle(90)
    set_angle(180)
