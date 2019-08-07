'''from spotify_local import SpotifyLocal

with SpotifyLocal() as s:
        print("Conectado!")
'''
from time import sleep
bashCommand = "spotify  --uri='spotify:track:03bYLN5H3OjZ6CIpBcd4W3'"
bashCommand2 = "dbus-send --print-reply --dest=org.mpris.MediaPlayer2.spotify /org/mpris/MediaPlayer2 org.mpris.MediaPlayer2.Player.Play"

import subprocess
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

sleep(3)
process2 = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE)
output, error = process2.communicate()

