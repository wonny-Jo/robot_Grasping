__author__ = "Morten Lind, Olivier Roulet-Dubonnet"
__copyright__ = "Copyright 2011, NTNU/SINTEF Raufoss Manufacturing AS"
__credits__ = ["Morten Lind, Olivier Roulet-Dubonnet"]
__license__ = "GPLv3"

import logging
import socket
import struct
import time
import threading
from copy import deepcopy

import numpy as np

import math3d as m3d

class FTSensorMonitor(threading.Thread):

    def __init__(self, ftHost):
        threading.Thread.__init__(self)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.daemon = True
        self._stop_event = True
        self._dataEvent = threading.Condition()
        self._dataAccess = threading.Lock()
        self._rtSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._rtSock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._urHost = ftHost

        self._timestamp = None
        self._ctrlTimestamp = None
        self._ftdata = None

        self.__recvTime = 0
        self._last_ctrl_ts = 0
        self._buffering = False
        self._buffer_lock = threading.Lock()
        self._buffer = []
        self._csys = None
        self._csys_lock = threading.Lock()

    def set_csys(self, csys):
        with self._csys_lock:
            self._csys = csys

    def __recv_bytes(self, nBytes):
        ''' Facility method for receiving exactly "nBytes" bytes from
        the robot connector socket.'''
        # Record the time of arrival of the first of the stream block
        recvTime = 0
        pkg = b''
        while len(pkg) < nBytes:
            pkg += self._rtSock.recv(nBytes)
            if recvTime == 0:
                recvTime = time.time()
        self.__recvTime = recvTime
        return pkg

    def wait(self):
        with self._dataEvent:
            self._dataEvent.wait()

    def get_fts_data(self):
        with self._dataAccess:
            ftdata = self._ftdata
            return ftdata
    getFTForce = get_fts_data

    def __recv_rt_data(self):
        temp = self.__recv_bytes(150)

        with self._dataAccess:
            tmp_bytes = temp.decode('utf-8')

            str1 = tmp_bytes.rfind(')')
            str2 = tmp_bytes.rfind('(', 1, str1)

            tmp = tmp_bytes[str2+1:str1]
            e = [float(x) for x in tmp.split(',')]

            self._ftdata = np.array([e[0], e[1], e[2], e[3], e[4], e[5]])

    def start_buffering(self):
        """
        start buffering all data from controller
        """
        self._buffer = []
        self._buffering = True

    def stop_buffering(self):
        self._buffering = False

    def try_pop_buffer(self):
        """
        return oldest value in buffer
        """
        with self._buffer_lock:
            if len(self._buffer) > 0:
                return self._buffer.pop(0)
            else:
                return None

    def pop_buffer(self):
        """
        return oldest value in buffer
        """
        while True:
            with self._buffer_lock:
                if len(self._buffer) > 0:
                    return self._buffer.pop(0)
            time.sleep(0.001)

    def get_buffer(self):
        """
        return a copy of the entire buffer
        """
        with self._buffer_lock:
            return deepcopy(self._buffer)

    def stop(self):
        #print(self.__class__.__name__+': Stopping')
        self._stop_event = True

    def close(self):
        self.stop()
        self.join()

    def run(self):
        self._stop_event = False
        self._rtSock.connect((self._urHost, 63351))
        while not self._stop_event:
            self.__recv_rt_data()
        self._rtSock.close()