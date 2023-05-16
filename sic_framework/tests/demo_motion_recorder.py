import time

from sic_framework.devices.common_naoqi.common_naoqi_motion import NaoqiMotionTools
from sic_framework.devices.common_naoqi.naoqi_motion_recorder import NaoMotionRecorder, StartRecording, StopRecording, \
    PlayRecording, NaoMotionRecorderConf

conf = NaoMotionRecorderConf()
recorder = NaoMotionRecorder("192.168.0.151", conf=conf)

recorder.request(StartRecording(["Body"]))

# recorder.request(StartRecording(["HeadYaw", "HeadPitch"]))
print("Start moving the robot!")

record_time = 5

time.sleep(record_time)

recording = recorder.request(StopRecording())
print("Done")

print(recording)

print("Replaying action")

recorder.request(PlayRecording(recording))

time.sleep(record_time)
print("end")
