import cv2

def rtsp_stream( ip: str, channel: str, username: str = "admin", password: str = "itinl123", port: int = 554):
    rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/{channel}"
    cam = cv2.VideoCapture(rtsp_url)

    if not cam.isOpened():
        raise ConnectionError(f"Gagal terhubung ke RTSP stream: {rtsp_url}")
    
    return cam