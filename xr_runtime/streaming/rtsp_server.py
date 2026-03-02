import sys
# Configure Loguru FIRST, before any imports
from loguru import logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr, 
    level="INFO", 
    filter=lambda record: not (
        record["name"].startswith("xr_service_library") and 
        record["level"].name == "WARNING"
    )
)

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib, GObject
import cv2
import argparse
import threading
import time
import numpy as np

# Add parent directory to path to import xr_service_library
sys.path.append('../../')
# NOW import XRServiceConnection after logger is configured
from xr_service_library import XRServiceConnection


class CustomRTSPMediaFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, frame_provider, width=1920, height=1080, framerate=30):
        GstRtspServer.RTSPMediaFactory.__init__(self)
        self.frame_provider = frame_provider
        self.number_frames = 0
        self.fps = framerate
        self.width = width
        self.height = height
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = (
            f"appsrc name=source is-live=true block=true format=GST_FORMAT_TIME "
            f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={framerate}/1 ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            f"x264enc tune=zerolatency bitrate=2000 speed-preset=superfast key-int-max={framerate} ! "
            "rtph264pay name=pay0 pt=96"
        )
        self.set_launch(self.launch_string)

    def do_create_element(self, url):
        pipeline = Gst.parse_launch(self.launch_string)
        source = pipeline.get_by_name("source")
        source.connect("need-data", self.on_need_data)
        return pipeline

    def on_need_data(self, src, length):
        frame = self.frame_provider.get_frame()
        if frame is not None:
            # Resize frame if dimensions don't match
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
                
            data = frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            
            # Set buffer timestamp and duration
            buf.pts = buf.dts = int(self.number_frames * self.duration)
            buf.duration = int(self.duration)
            self.number_frames += 1
            
            # Push buffer to the pipeline
            retval = src.emit("push-buffer", buf)
            if retval != Gst.FlowReturn.OK:
                logger.warning(f"Push buffer returned {retval}")


class XRRtspServer:
    def __init__(self, socket_path='/tmp/xr_service.sock', port=8554, path='/xr_stream', width=1920, height=1080, framerate=30):
        """Initialize RTSP server that streams frames from XR Service

        Args:
            socket_path (str): Path to the XR Service Unix socket
            port (int): RTSP server port
            path (str): RTSP stream path
            width (int): Output video width
            height (int): Output video height

        """
        self.socket_path = socket_path
        self.port = port
        self.path = path
        self.width = width
        self.height = height
        self.client = XRServiceConnection(socket_path=socket_path)
        self.framerate = framerate
        self.client.connect()
        self.is_running = False
        self.latest_frame = None
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Create RTSP server
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service(str(port))
        
        # Create custom factory
        self.factory = CustomRTSPMediaFactory(self, width=width, height=height, framerate=self.framerate)
        
        # Set up factory to handle dynamic media
        self.factory.set_shared(True)
        
        # Connect factory to path
        self.server.get_mount_points().add_factory(path, self.factory)
        
        logger.info(f"RTSP server initialized on rtsp://localhost:{port}{path}")

    def _extract_numpy_frame(self, frame_data):
        """Extract a numpy array from various possible frame_data structures."""
        if hasattr(frame_data, 'frame'):
            frame_obj = frame_data.frame
            if hasattr(frame_obj, 'data') and hasattr(frame_obj, 'dtype') and hasattr(frame_obj, 'shape'):
                return np.frombuffer(frame_obj.data, dtype=frame_obj.dtype).reshape(frame_obj.shape)
            else:
                logger.warning(f"Frame object missing required attributes: {dir(frame_obj)}")
                return None
        elif isinstance(frame_data, np.ndarray):
            return frame_data
        elif isinstance(frame_data, dict) and 'image' in frame_data:
            return frame_data['image']
        else:
            logger.warning(f"Unexpected frame data type: {type(frame_data)}")
            return None
    
    def get_frame(self):
        """Return the latest frame for the pipeline"""
        return self.latest_frame
    
    def _frame_fetcher(self):
        """Thread function to continuously fetch frames from XR Service"""
        logger.info("Frame fetcher thread started")
        no_frame_count = 0
        while self.is_running:
            try:
                frame_data = self.client.get_latest_frame()
                if frame_data is not None:
                    frame = self._extract_numpy_frame(frame_data)
                    if frame is not None:
                        self.latest_frame = frame
                else:
                    no_frame_count += 1
                    if no_frame_count % 100 == 1:
                        logger.debug("No frames available yet")
            except Exception as e:
                logger.error(f"Error fetching frame: {e}")
                logger.error(f"Frame data type: {type(frame_data) if 'frame_data' in locals() else 'undefined'}")
            time.sleep(0.01)  # Small sleep to avoid overloading the CPU
    
    def start(self):
        """Start the RTSP server and frame fetcher thread"""
        # Start the server
        self.server_id = self.server.attach(None)
        
        # Start the main GLib loop in a separate thread
        self.loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self.loop.run)
        self.loop_thread.daemon = True
        self.loop_thread.start()
        
        # Start frame fetcher thread
        self.is_running = True
        self.fetcher_thread = threading.Thread(target=self._frame_fetcher)
        self.fetcher_thread.daemon = True
        self.fetcher_thread.start()
        
        logger.info("RTSP server started")
    
    def stop(self):
        """Stop the RTSP server and frame fetcher thread"""
        self.is_running = False
        
        # Wait for fetcher thread to complete
        if hasattr(self, 'fetcher_thread') and self.fetcher_thread.is_alive():
            self.fetcher_thread.join(timeout=1.0)
        
        # Stop GLib main loop
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.quit()
        
        # Wait for loop thread to complete
        if hasattr(self, 'loop_thread') and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=1.0)
        
        logger.info("RTSP server stopped")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RTSP server for XR Service video streaming')
    parser.add_argument('--socket-path', type=str, required=True,
                      help='Path to the Unix domain socket for the XR Service (default: /tmp/xr_service.sock)')
    parser.add_argument('--port', type=int, required=True,
                      help='RTSP server port (default: 8554)')
    parser.add_argument('--path', type=str, default='/xr_stream',
                      help='RTSP stream path (default: /xr_stream)')
    parser.add_argument('--width', type=int, default=1920,
                      help='Output video width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                      help='Output video height (default: 1080)')
    parser.add_argument('--framerate', type=int, default=30,
                      help='Output video framerate (default: 30)')

    parser.add_argument('--log-level', type=str, default='INFO',
                      help='Log level (default: INFO)')
    args = parser.parse_args()

    # Configure logging
    logger.configure(handlers=[{"sink": sys.stderr, "level": args.log_level}])

    # Create and start RTSP server
    server = XRRtspServer(
        socket_path=args.socket_path,
        port=args.port,
        path=args.path,
        width=args.width,
        height=args.height,
        framerate=args.framerate
        )   
    
    try:
        server.start()
        logger.info(f"RTSP stream available at rtsp://localhost:{args.port}{args.path}")
        logger.info(f"Streaming resolution: {args.width}x{args.height}")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Error in main thread: {e}")
    finally:
        server.stop()

if __name__ == '__main__':
    main()