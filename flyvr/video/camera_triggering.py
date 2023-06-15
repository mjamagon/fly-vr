
import time
import numpy as np
import PySpin
from imageio_ffmpeg import write_frames, get_ffmpeg_exe, get_ffmpeg_version
from flyvr.common import SharedState, BACKEND_CAMERA
from flyvr.common.build_arg_parser import setup_logging
from flyvr.common.ipc import PlaylistReciever, CommonMessages
import cv2
import os
import threading
from threading import Thread

class Recorder(threading.Thread):
    def __init__(self,camera,fileName:str,camName:str):
        Thread.__init__(self)
        self.daemon = True
        self.fileName = fileName
        self.camName = camName # name of camera
        self.camera = camera 
        self.isAlive = True
        self._lock = threading.Lock()

       # Initialize recorder 
        self._open_recorder()

    def _open_recorder(self):
        # self.recorder.Open(self.fileName,self.option) 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        savePath = os.path.join(self.fileName,f'{self.camName}.mp4')
        self.vidWrite = cv2.VideoWriter(savePath,fourcc,isColor=False,frameSize=(560,560),fps=75)

    def _close_recorder(self):
        self.isAlive = False
        print('shutting down camera recording...')
        shutdownSuccess = False 
        while not shutdownSuccess:
            try:
                self.vidWrite.release()
                shutdownSuccess = True
            except: 
                pass

    def run(self):
        while self.isAlive: 
            # print('doing',flush=True)
            try:
                with self._lock:
                    image = self.camera.GetNextImage()
                    image = image.Convert(PySpin.PixelFormat_Mono8,PySpin.HQ_LINEAR)
                    imageArray = image.GetNDArray() # get data from pointer as numpy array
                    self.vidWrite.write(imageArray)
            except Exception as e:
                print(e,flush=True)

class Camera:
    def __init__(self,camera,nodemap,camName:str):
        self.camera = camera # pyspin camera object 
        self.nodemap = nodemap
        self.recorder = None
        self.camName = camName # name of camera 

    def start_recording(self,fileName):
        print('starting camera recording...',flush=True)

        # Acquire and save images 
        self.camera.BeginAcquisition()
        recorder = Recorder(camera=self.camera,fileName=fileName,camName=self.camName)
        recorder.start()
        self.recorder = recorder

        
    def configure_trigger(self):
        """
        This function configures the camera to use a trigger. First, trigger mode is
        set to off in order to select the trigger source. Once the trigger source
        has been selected, trigger mode is then enabled, which has the camera
        capture only a single image upon the execution of the chosen trigger.
        :param cam: Camera to configure trigger for.
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool

        %% Notes %%
        Modified from https://github.com/nimble00/PTGREY-cameras-with-python/blob/master/Trigger_QuickSpin.py
        """
        cam = self.camera
        result = True
        CHOSEN_TRIGGER = 1

        if CHOSEN_TRIGGER == 1:
            print ("Software trigger chosen...")
        elif CHOSEN_TRIGGER == 2:
            print ("Hardware trigger chosen...")
        try:

            # Ensure trigger mode off
            # The trigger must be disabled in order to configure whether the source
            # is software or hardware
            nodemap = cam.GetNodeMap()
            node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerMode"))

            if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
                print ("Unable to disable trigger mode (node retrieval). Aborting...")
                return False

            node_trigger_mode_off = node_trigger_mode.GetEntryByName("Off")
            if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
                print ("Unable to disable trigger mode (enum entry retrieval). Aborting...")
                return False

            node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

            print ("Trigger mode disabled...")

            # Select trigger source
            # The trigger source must be set to hardware or software while trigger
            # mode is off.
            node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerSource"))
            if not PySpin.IsAvailable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
                print ("Unable to get trigger source (node retrieval). Aborting...")
                return False

            if CHOSEN_TRIGGER == 1:
                node_trigger_source_software = node_trigger_source.GetEntryByName("Software")
                if not PySpin.IsAvailable(node_trigger_source_software) or not PySpin.IsReadable(
                        node_trigger_source_software):
                    print ("Unable to set trigger source (enum entry retrieval). Aborting...")
                    return False
                node_trigger_source.SetIntValue(node_trigger_source_software.GetValue())

            elif CHOSEN_TRIGGER == 2:
                node_trigger_source_hardware = node_trigger_source.GetEntryByName("Line0")
                if not PySpin.IsAvailable(node_trigger_source_hardware) or not PySpin.IsReadable(
                        node_trigger_source_hardware):
                    print ("Unable to set trigger source (enum entry retrieval). Aborting...")
                    return False
                node_trigger_source.SetIntValue(node_trigger_source_hardware.GetValue())

            # Turn trigger mode on
            # Once the appropriate trigger source has been set, turn trigger mode
            # on in order to retrieve images using the trigger.
            node_trigger_mode_on = node_trigger_mode.GetEntryByName("On")
            if not PySpin.IsAvailable(node_trigger_mode_on) or not PySpin.IsReadable(node_trigger_mode_on):
                print ("Unable to enable trigger mode (enum entry retrieval). Aborting...")
                return False

            node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())
            print ("Trigger mode turned back on...")

        except PySpin.SpinnakerException as ex:
            print ("Error: %s" % ex)
            return False

        return result

    def execute_trigger(self):
        nodemap = self.nodemap
        node_softwaretrigger_cmd = PySpin.CCommandPtr(nodemap.GetNode("TriggerSoftware"))

        if not PySpin.IsAvailable(node_softwaretrigger_cmd) or not PySpin.IsWritable(node_softwaretrigger_cmd):
            print("Unable to execute trigger. Aborting...")
            return False

        time.sleep(2)
        node_softwaretrigger_cmd.Execute()
        print('camera successfully triggered')    
        time.sleep(2)

    def record_video(self):
        # Find the camera and initialize
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        cam = cam_list.GetBySerial(self.options['sn'])
        cam.Init()

        # Get camera nodemap and TL device
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        nodemap = cam.GetNodeMap()
        import pdb; pdb.set_trace()
        return

if __name__ == "__main__":
    options = {
        'snPrimary':'15058177',
        'snSecondary':'17215641',
        'fileName':"D:\max_flyvr\camera_sync"
    }

    # Get primary camera 
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam1 = cam_list.GetBySerial(options['snPrimary'])
    cam1.Init()
    nodemap1 = cam1.GetNodeMap()
    primaryCam = Camera(cam1,nodemap1) 
    primaryCam.configure_trigger()
    primaryCam.execute_trigger()

    # Get secondary camera
    cam2 = cam_list.GetBySerial(options['snSecondary'])
    cam2.Init() 
    nodemap2 = cam2.GetNodeMap()
    secondaryCam = Camera(cam2,nodemap2) 

    # Record for five seconds 
    secondaryCam.start_recording(fileName=options['fileName'])
    t0 = time.time()
    t1 = time.time()
    while t1-t0<5:
        print(t1-t0)
        t1 = time.time()
    secondaryCam.recorder._close_recorder()
    print('Done!')


