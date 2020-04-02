from channels.generic.websocket import WebsocketConsumer
import json
import sys
import time
from threading import Thread
import subprocess
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import re
import math
from shutil import copyfile, rmtree
import os
import cv2
from datetime import datetime
import datetime
import os.path
from tkinter.filedialog import askopenfilename
import youtube_dl
from queue import Queue
from contextlib import contextmanager
from urllib.error import HTTPError
import atexit
from django.conf import settings


TEMP_FOLDER = "static/media/process/videocuts_tmp"
YSTAMP = "Youtube_{}_tmp".format(datetime.datetime.now().strftime("%H%M%S"))
YTUBE_FOLDER = "static/media/process/{}".format(YSTAMP)
isCanceld = False
pth = ''
times, timem, timeh = 0, 0, 0
stpTimer = False
gain = 1.2


def inputToOutputFilename(filename):
    currentDT = datetime.datetime.now()
    dotIndex = filename.rfind(".")
    return str(filename[:dotIndex] + "_videocuts_" + currentDT.strftime("%H%M%S") + filename[dotIndex:])


def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv, -minv)


def copyFrame(inputFrame, outputFrame):
    src = TEMP_FOLDER + "/frame{:06d}".format(inputFrame + 1) + ".jpg"
    dst = TEMP_FOLDER + "/newFrame{:06d}".format(outputFrame + 1) + ".jpg"
    if not os.path.isfile(src):
        return False
    copyfile(src, dst)
    if outputFrame % 20 == 19:
        str(outputFrame + 1) + " time-altered frames saved."
    return True


def createPath(s):
    try:
        os.mkdir(s)
    except OSError:
        sys.exit()  # Check for error


def deletePath(s):  # Dangerous! Watch out!
    try:
        rmtree(s, ignore_errors=True)
    except OSError:
        sys.exit()  # Check for error


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def get_size(filename):
    size = int(os.path.getsize(filename)) / 1000000
    return str(size)


@contextmanager
def change_dir(destination):
    cwd = os.getcwd()
    try:
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)


class Process:

    def get_length(self, filename):
        try:
            cap = cv2.VideoCapture(filename)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            minutes = int(duration / 60)
            seconds = int(duration % 60)
            return str(minutes) + ' min ' + str(seconds) + ' sec'
        except Exception as e:
            self.cancel()
            self.onError()
            self.timer_queue.put(' Processing Video Failed! ')
            self.timer_queue.put('Message Error.' + str(e))
            sys.exit()
        except HTTPError as err:
            if err.code == 500:
                self.timer_queue.put(' Processing Video Failed! ')
                self.timer_queue.put('Message Error.' + str(err))
                self.cancel()
                sys.exit()

    def __init__(self, video_url, silence_threshold, silent_speed, frame_margin, frame_quality, play_speed,
                 frame_rate, resolution):
        self.video_url = video_url
        self.silence_threshold = silence_threshold
        self.silent_speed = silent_speed
        self.frame_margin = frame_margin
        self.frame_quality = frame_quality
        self.play_speed = play_speed
        self.frame_rate = frame_rate
        self.resol = resolution
        self.output_parameters = list()
        self.timer_queue = Queue()
        self.output_queue = Queue()
        self.stop = False

    def timer(self):
        global times, timem
        while True:
            if stpTimer:
                break
            if isCanceld:
                break
            if self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                break
            if times == 60:
                times = 0
                timem = timem + 1
            self.elapsed_timer = str('Elapsed Processing Time: ' + str(timem) + ' min ' + str(times) + ' sec')
            self.timer_queue.put(self.elapsed_timer)
            time.sleep(1)
            times = times + 1

    def newFileChecker(self, path=None):
        global gain
        info0 = str(self.origin_video_length).split(' min ')
        olddurSec = int(info0[0]) * 60 + int(info0[1].split(' sec')[0])
        while True:
            time.sleep(2)
            if stpTimer:
                break
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                break
            try:
                inf = str(self.elapsed_timer).split('Elapsed Processing Time: ')[1].split(' min ')
                new = int(inf[0]) * 60 + int(inf[1].split(' sec')[0])
            except Exception:
                new = 0

    def downloadFile(self, url):
        global gain, isCanceld
        try:
            with youtube_dl.YoutubeDL() as ydl:
                minfo = ydl.extract_info(url, download=False)
                video_title = minfo.get('title', None)
                title = ''.join(e for e in video_title if (e.isalnum() or e.isspace()))
                self.timer_queue.put(' Step 0 - Getting Youtube Video Duration ... ')
                if self.resol == 'Default':
                    myFormat = 'best'
                else:
                    myFormat = 'bestvideo[ext=mp4, height<=?' + str(self.resol).split('p')[0] + ']‌​+bestaudio[ext=m4a]'
                ydl_opts = {
                    'format': myFormat,  # f'bestvideo[ext=mp4{qlt}]‌​+bestaudio[ext=m4a]'
                    'outtmpl': f'static/media/process/{YSTAMP}/{title}.%(ext)s',
                }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                minfo = ydl.extract_info(url, download=False)
                filename = ydl.prepare_filename(minfo)
                dur = minfo['duration']
                if dur >= 60:
                    str_min = dur / 60
                else:
                    str_min = 0
                str_sec = dur % 60
                self.origin_video_length = str(str_min).split('.')[0] + ' min ' + str(str_sec) + ' sec'
                self.output_parameters.append(self.origin_video_length)
                self.output_queue.put('ovl' + ' ' + self.origin_video_length)
                Thread(target=self.newFileChecker).start()
                if not isCanceld or self.stop:
                    self.timer_queue.put(' Step 0 - Downloading Youtube Video in Progress ... ')
                ydl.download([url])

                self.origin_video_size = get_size(filename) + ' MB'
                self.output_parameters.append(self.origin_video_size)
                self.output_queue.put('ovs' + ' ' + self.origin_video_size)

        except Exception as e:
            self.cancel()
            self.onError()
            self.timer_queue.put('Step 0 - Downloading Youtube Video Failed !')
            if str(e).__contains__('The system cannot find the file specified'):
                self.timer_queue.put(
                    'downloading , Requested formats are incompatible\nPlease Choose The Default Quality')
            else:
                self.timer_queue.put('downloading .' + str(e))
            sys.exit()
        return filename

    def callback2(self, sv, lbl, vld, name):
        try:
            if vld[0] <= float(sv.get()) <= vld[1]:
                lbl.config(text='')
            else:
                lbl.config(text=f'{name} Value must be \nin this range ({vld[0]} to {vld[1]})', anchor='w')
        except Exception:
            lbl.config(text=f'{name} Value must be \nin this range ({vld[0]} to {vld[1]})', anchor='w')

    def process(self):
        folder_pattern = re.compile(r'Youtube_\d+_tmp')
        global isCanceld
        file_path = 'static/media/video_input'
        try:
            if os.path.exists(TEMP_FOLDER):
                deletePath(TEMP_FOLDER)
            folders = [folder for folder in os.listdir("static/media/process") if folder_pattern.match(folder)]
            for folder in folders:
                deletePath("static/media/process/" + folder)
            Thread(target=self.timer).start()
            global gain
            gain = 1.2
            self.new_video_size = 'N/A'
            self.new_video_length = 'N/A'
            Extras = ""
            frameRate = float(60)
            SAMPLE_RATE = int(self.frame_rate)
            SILENT_THRESHOLD = float(self.silence_threshold)
            FRAME_SPREADAGE = int(self.frame_margin)
            NEW_SPEED = [float(self.silent_speed), float(self.play_speed)]
            gain = 0.6
            INPUT_FILE = ""
            re_dir = '../video_input'
            if self.video_url == " ":
                if os.listdir(file_path):
                    with change_dir('static/media/process'):
                        os.mkdir(f'{YSTAMP}')
                        for file in os.listdir(re_dir):
                            copyfile(os.path.join(re_dir, file), f'{YSTAMP}/' + file)

                    for file in os.listdir(file_path):
                        dir_file = file_path + '/' + file
                        if os.path.splitext(dir_file)[1].lower() in ('.mp4', '.mov', '.mpeg', '.wmv'):
                            INPUT_FILE = f'static/media/process/{YSTAMP}/' + file
                            self.origin_video_size = get_size(INPUT_FILE)
                            self.output_queue.put('ovs' + ' ' + self.origin_video_size)
                            self.origin_video_length = self.get_length(INPUT_FILE)
                            self.output_queue.put('ovl' + ' ' + self.origin_video_length)
                        else:
                            self.timer_queue.put('Please choose an appropriate video file')
                            INPUT_FILE = None
                        os.remove(os.path.join("static/media/video_input", file))
                else:
                    INPUT_FILE = None
            else:
                INPUT_FILE = self.downloadFile(str(self.video_url))

            FRAME_QUALITY = self.frame_quality

            assert INPUT_FILE is not None, "You did not specify an input file.  You must specify an input file without spaces."

            OUTPUT_FILE = inputToOutputFilename(INPUT_FILE)

            AUDIO_FADE_ENVELOPE_SIZE = 400  # smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)
            createPath(TEMP_FOLDER)

            global dir
            dir = os.getcwd()
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return

            self.timer_queue.put(' Step 1 - Frame quality has been assessed and is processing ')
            cmdary = [resource_path('ffmpeg.exe'), "-i", INPUT_FILE, '-qscale:v', str(
                FRAME_QUALITY), TEMP_FOLDER + "/frame%06d.jpg", '-hide_banner']
            subprocess.call(cmdary, cwd=dir, shell=True)
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return
            self.timer_queue.put(' Step 1 - Frame quality processing has successfully completed ')

            time.sleep(2)
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return
            self.timer_queue.put(' Step 2 - Sample Rate has been assessed and is processing ')
            cmdary = [resource_path('ffmpeg.exe'), "-i", INPUT_FILE, '2>&1', '-ab', '160k', '-ac', '2', '-ar',
                      str(SAMPLE_RATE), '-vn', TEMP_FOLDER + "/audio.wav"]
            subprocess.call(cmdary, cwd=dir, shell=True)

            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return

            self.timer_queue.put(' Step 2 - Sample Rate processing has successfully completed ')
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return

            time.sleep(2)
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return

            self.timer_queue.put(' Step 3 - Video Frames are processing. This might take a while... ')
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return

            cmdary = [resource_path('ffmpeg.exe'), "-i", INPUT_FILE, '2>&1']
            open(TEMP_FOLDER + "/params.txt", "w")
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return

            subprocess.call(cmdary, cwd=dir, shell=True)
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return

            self.timer_queue.put(' Step 3 - Video Frames processing has successfully completed ')
            time.sleep(2)
            sampleRate, audioData = wavfile.read(TEMP_FOLDER + "/audio.wav")
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return
            audioSampleCount = audioData.shape[0]
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return
            maxAudioVolume = getMaxVolume(audioData)
            cap = cv2.VideoCapture(INPUT_FILE)
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return
            fps = cap.get(cv2.CAP_PROP_FPS)
            f = open(TEMP_FOLDER + "/params.txt", 'r+')
            pre_params = f.read()
            f.close()
            params = pre_params.split('\n')
            for line in params:
                m = re.search(' ([0-9]*.[0-9]*) fps,', line)
                if m is None:
                    frameRate = float(fps)
                if m is not None:
                    frameRate = float(m.group(1))

            samplesPerFrame = sampleRate / frameRate
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return
            audioFrameCount = int(math.ceil(audioSampleCount / samplesPerFrame))
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return
            hasLoudAudio = np.zeros(audioFrameCount)

            for i in range(audioFrameCount):
                start = int(i * samplesPerFrame)
                end = min(int((i + 1) * samplesPerFrame), audioSampleCount)
                audiochunks = audioData[start:end]
                maxchunksVolume = float(getMaxVolume(audiochunks)) / maxAudioVolume
                if maxchunksVolume >= SILENT_THRESHOLD:
                    hasLoudAudio[i] = 1
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return

            chunks = [[0, 0, 0]]
            shouldIncludeFrame = np.zeros(audioFrameCount)
            for i in range(audioFrameCount):
                start = int(max(0, i - FRAME_SPREADAGE))
                end = int(min(audioFrameCount, i + 1 + FRAME_SPREADAGE))
                shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end])
                if i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i - 1]:  # Did we flip?
                    chunks.append([chunks[-1][1], i, shouldIncludeFrame[i - 1]])
                if isCanceld or self.stop:
                    deletePath(YTUBE_FOLDER)
                    deletePath(TEMP_FOLDER)
                    return

            chunks.append([chunks[-1][1], audioFrameCount, shouldIncludeFrame[i - 1]])
            chunks = chunks[1:]

            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return

            outputAudioData = np.zeros((0, audioData.shape[1]))
            outputPointer = 0
            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return

            lastExistingFrame = None
            for chunk in chunks:
                audioChunk = audioData[int(chunk[0] * samplesPerFrame):int(chunk[1] * samplesPerFrame)]

                sFile = TEMP_FOLDER + "/tempStart.wav"
                eFile = TEMP_FOLDER + "/tempEnd.wav"
                wavfile.write(sFile, SAMPLE_RATE, audioChunk)

                if isCanceld or self.stop:
                    deletePath(YTUBE_FOLDER)
                    deletePath(TEMP_FOLDER)
                    return
                with WavReader(sFile) as reader:
                    with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
                        tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
                        tsm.run(reader, writer)
                _, alteredAudioData = wavfile.read(eFile)
                leng = alteredAudioData.shape[0]
                endPointer = outputPointer + leng
                outputAudioData = np.concatenate((outputAudioData, alteredAudioData / maxAudioVolume))
                if isCanceld or self.stop:
                    deletePath(YTUBE_FOLDER)
                    deletePath(TEMP_FOLDER)
                    return

                # outputAudioData[outputPointer:endPointer] = alteredAudioData/maxAudioVolume

                # smooth out transitiion's audio by quickly fading in/out

                if leng < AUDIO_FADE_ENVELOPE_SIZE:
                    outputAudioData[
                    outputPointer:endPointer] = 0  # audio is less than 0.01 sec, let's just remove it.
                else:
                    premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE) / AUDIO_FADE_ENVELOPE_SIZE
                    mask = np.repeat(premask[:, np.newaxis], 2, axis=1)  # make the fade-envelope mask stereo
                    outputAudioData[outputPointer:outputPointer + AUDIO_FADE_ENVELOPE_SIZE] *= mask
                    outputAudioData[endPointer - AUDIO_FADE_ENVELOPE_SIZE:endPointer] *= 1 - mask
                if isCanceld or self.stop:
                    deletePath(YTUBE_FOLDER)
                    deletePath(TEMP_FOLDER)
                    return

                startOutputFrame = int(math.ceil(outputPointer / samplesPerFrame))
                endOutputFrame = int(math.ceil(endPointer / samplesPerFrame))
                for outputFrame in range(startOutputFrame, endOutputFrame):
                    inputFrame = int(chunk[0] + NEW_SPEED[int(chunk[2])] * (outputFrame - startOutputFrame))
                    didItWork = copyFrame(inputFrame, outputFrame)
                    if didItWork:
                        lastExistingFrame = inputFrame
                    else:
                        copyFrame(lastExistingFrame, outputFrame)
                    if isCanceld or self.stop:
                        deletePath(YTUBE_FOLDER)
                        deletePath(TEMP_FOLDER)
                        return

                outputPointer = endPointer

            wavfile.write(TEMP_FOLDER + "/audioNew.wav", SAMPLE_RATE, outputAudioData)
            '''
            outputFrame = math.ceil(outputPointer/samplesPerFrame)
            for endGap in range(outputFrame,audioFrameCount):
                copyFrame(int(audioSampleCount/samplesPerFrame)-1,endGap)
            '''

            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return
            self.timer_queue.put(' Step 4 - Finalizing.... Please wait')
            cmdary = [resource_path('ffmpeg.exe'), '-framerate', str(frameRate), "-i",
                      TEMP_FOLDER + "/newFrame%06d.jpg", '-i', TEMP_FOLDER + "/audioNew.wav", '-strict', '-2' + str(
                    Extras), OUTPUT_FILE]
            subprocess.call(cmdary, cwd=dir, shell=True)

            if isCanceld or self.stop:
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
                return
            self.timer_queue.put(' Video processing finished successfully.')

            deletePath(TEMP_FOLDER)
            path = os.path.dirname(INPUT_FILE)
            global stpTimer
            stpTimer = True
            self.new_video_size = get_size(OUTPUT_FILE) + ' MB'
            self.output_parameters.append(self.new_video_size)
            self.output_queue.put('nvs' + ' ' + self.new_video_size)
            self.new_video_length = str(self.get_length(OUTPUT_FILE))
            self.output_parameters.append(self.new_video_length)
            self.output_queue.put('nvl' + ' ' + self.new_video_length)
            file_name = str(OUTPUT_FILE)
            self.output_queue.put(file_name)

        except Exception as e:
            self.cancel()
            self.onError()
            self.timer_queue.put(' Processing Video Failed! ')
            if str(e) != 'main thread is not in main loop':
                self.timer_queue.put(' message.' + str(e))
            deletePath(TEMP_FOLDER)
            sys.exit()

    def main(self):
        global isCanceld, stpTimer, times, timem
        try:
            times = 0
            timem = 0
            isCanceld = False
            stpTimer = False
            Thread(target=self.process).start()
        except Exception as e:
            self.cancel()
            self.onError()
        except HTTPError as err:
            if err.code == 500:
                self.cancel()
                sys.exit()

    def getFilePath(self):
        global pth
        pth1 = str(askopenfilename(initialdir=os.getcwd()))
        pthext = pth1.split('/')[-1]
        pth = ''.join(e for e in pthext if (e.isalnum() or e.isspace() or e == '/' or e == '.' or e == ':'))
        pth = pth1.replace(pthext, pth)
        os.rename(pth1, pth)

    def cancel(self):
        global isCanceld
        isCanceld = True
        while os.path.exists(TEMP_FOLDER):
            deletePath(TEMP_FOLDER)
        deletePath(TEMP_FOLDER)

    def onError(self):
        global isCanceld
        isCanceld = True


def run_process(url, st, ss, fm, fq, ps, fr, qlty):
    process = Process(url, st, ss, fm, fq, ps, fr, qlty)
    return process


def exit_handler():
    if os.path.join(settings.BASE_DIR, TEMP_FOLDER):
        deletePath(os.path.join(settings.BASE_DIR, TEMP_FOLDER))


class RunProcess(WebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process = run_process(**self.scope['url_route']['kwargs'])

    def websocket_connect(self, event):
        self.accept()

        def output_thread():
            while True:
                time = self.process.timer_queue.get()
                self.send(time)
                if self.process.stop:
                    break
                if time in (' Video processing finished successfully.',):
                    break
            while True:
                parm = self.process.output_queue.get()
                self.send(parm)
                if self.process.stop:
                    break
                if parm[:3].lower() in ('nvl',):
                    break

        if not self.process.stop:
            Thread(target=output_thread, daemon=True).start()

    def websocket_receive(self, event):
        json_data = json.dumps({
            "type": "websocket.accept",
        })
        text = event.get('text', None)
        if str(text) == 'stop':
            self.process.stop = True
            if os.path.exists(TEMP_FOLDER):
                deletePath(YTUBE_FOLDER)
                deletePath(TEMP_FOLDER)
        else:
            self.process.main()

    def websocket_disconnect(self, event):
        global isCanceld
        isCanceld = True
        self.process.stop = True
        if os.path.exists(TEMP_FOLDER):
            deletePath(TEMP_FOLDER)
        deletePath(TEMP_FOLDER)


atexit.register(exit_handler)
