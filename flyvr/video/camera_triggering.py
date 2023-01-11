
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

def threaded(fn):
    def wrapper(*args, **kwargs):
        threading.Thread(target=fn, args=args, kwargs=kwargs).start()
    return wrapper

class Recorder:
    def __init__(self,camera,compressionType:int,fileName:str):
        assert compressionType in [0,1,2]
        self.recorder = PySpin.SpinVideo()
        self.fileName = fileName
        self.camera = camera 
        self.isAlive = True

        # # Set compression mode
        # if compressionType==0:
        #     self.option = PySpin.AVIOption()
        # elif compressionType==1: # mjpg compression
        #     self.option= PySpin.MJPGOption()
        #     self.option.quality=75
        # elif compressionType==2: # h264 compression
        #     self.option = PySpin.H264Option()


    def _open_recorder(self):
        # self.recorder.Open(self.fileName,self.option) 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        savePath = os.path.join(self.fileName,'secondary.mp4')
        self.vidWrite = cv2.VideoWriter(savePath,fourcc,frameSize=(560,560),fps=75)

    def _close_recorder(self):
        # self.recorder.Close()
        self.vidWrite.release()
        
    @threaded
    def acquire_images(self):
        # start AVI recorder 

        # import pdb; pdb.set_trace()
        self._open_recorder()

        while self.isAlive:
            image = self.camera.GetNextImage()
            image = image.Convert(PySpin.PixelFormat_Mono8,PySpin.HQ_LINEAR)
            imageArray = image.GetNDArray() # get data from pointer as numpy array
            # self.recorder.Append(image)
            self.vidWrite.write(imageArray)

        print('Closing recorder...')
        # self.recorder.close_recorder()   
        self.vidWrite.release()


class Camera_Server:
    def __init__(self,options:dict):
        assert 'snPrimary' and 'snSecondary' and 'fileName' in options
        self.options = options

    # def setup_camera(self,sn):
    #     '''Find camera'''
    #     system = PySpin.System.GetInstance()
    #     cam_list = system.GetCameras()
    #     cam = cam_list.GetBySerial(self.options['snPrimary'])
    #     return cam

    def acquire_images(self,recorder):
        # start AVI recorder 
        recorder = Recorder(compressionType=2,fileName=self.options['savePath'])
        recorder.open_recorder()

        while self.isAlive:
            image = self.secondaryCam.GetNextImage()
            image = image.Convert(PySpin.PixelFormat_Mono8,PySpin.HQ_LINEAR)
            # imageArray = image.GetNDArray() # get data from pointer as numpy array
            recorder.recorder.AVIappend(image)

        recorder.close_recorder()
       
    def run_server(self):
        # Find the primary camera and initialize 
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        primaryCam = cam_list.GetBySerial(self.options['snPrimary'])
        # primaryCam = self.setup_camera(self.options['snPrimary'])
        primaryCam.Init()
        self.primaryCam=primaryCam

        ''' %% Trigger the primary camera to initialize the recording %% '''
        # Trigger the primary camera
        _ = self.configure_trigger()

        # Get nodemap 
        nodemap = self.primaryCam.GetNodeMap()
    
        # Execute a software trigger 
        self.execute_trigger(nodemap)

        ''' %% Start recording from the secondary camera %% '''
        # First find and initialize the secondary camera 
        secondaryCam = cam_list.GetBySerial(self.options['snSecondary'])
        secondaryCam.Init()
        secondaryCam.BeginAcquisition()
        self.secondaryCam = secondaryCam

        # Acquire and save images 
        recorder = Recorder(camera=secondaryCam,fileName=self.options['fileName'],compressionType=2)
        recorder.acquire_images()
        # self.acquire_images()

        
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
        cam = self.primaryCam
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

    def execute_trigger(self,nodemap):
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
        # 'fileName':'test'
    }
    server =Camera_Server(options=options)
    server.run_server()
    # time.sleep(2)
    t0 = time.time()
    t1 = time.time()
    while t1-t0<10:
        t1 = time.time()
        print(t1-t0)
        continue
    server.recorder.isAlive = False