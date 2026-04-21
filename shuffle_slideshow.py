import os
import sys
import signal
import random
import time
import cv2
import numpy as np
import argparse
from geopy.geocoders import Nominatim
from exif import Image as ExifImage
from datetime import datetime
from PIL import Image as PILImage, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS

# Configuration
TIMER_DELAY = 10000  # ms for images
GEOLOCATOR = Nominatim(user_agent="shuffle_slideshow_tool_v15", timeout=10)
GEO_CACHE = {}

# Force X11 for OpenCV/Qt if on Linux
if os.environ.get('XDG_SESSION_TYPE') == 'wayland':
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

def cleanup_and_exit():
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, lambda sig, frame: cleanup_and_exit())

def format_date_str(date_input):
    if not date_input: return "Unknown Date"
    try:
        dt = datetime.strptime(str(date_input).split()[0], '%Y:%m:%d')
        return dt.strftime('%d.%m.%Y')
    except (ValueError, IndexError):
        try:
            dt = datetime.strptime(str(date_input).split()[0], '%Y-%m-%d')
            return dt.strftime('%d.%m.%Y')
        except (ValueError, IndexError):
            return str(date_input)

def get_geocoded_address(lat, lon):
    if lat is None or lon is None: return None
    coords = f"{lat}, {lon}"
    if coords in GEO_CACHE: return GEO_CACHE[coords]
    try:
        location = GEOLOCATOR.reverse(coords, language="en")
        if location:
            address = location.raw.get('address', {})
            city = address.get('city') or address.get('town') or address.get('village') or address.get('suburb') or "Unknown City"
            country = address.get('country', "Unknown Country")
            result = f"{city}, {country}"
            GEO_CACHE[coords] = result
            return result
    except Exception: pass
    return f"{lat:.4f}, {lon:.4f}"

def get_decimal_from_dms(dms, ref):
    if not dms: return None
    try:
        val = float(dms[0]) + float(dms[1])/60.0 + float(dms[2])/3600.0
        if ref and ref in ['S', 'W']: val = -val
        return val
    except (TypeError, ValueError, IndexError): return None

def get_gps_from_pillow(path):
    try:
        with PILImage.open(path) as img:
            exif = img._getexif()
            if not exif: return None, None
            gps = {GPSTAGS.get(t, t): exif[tag][t] for tag, val in exif.items() if TAGS.get(tag) == "GPSInfo" for t in exif[tag]}
            if "GPSLatitude" in gps:
                return get_decimal_from_dms(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N")), \
                       get_decimal_from_dms(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
    except Exception: pass
    return None, None

def get_image_metadata(path):
    raw_date = None
    try: raw_date = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d')
    except (OSError, ValueError): pass
    lat, lon = None, None
    try:
        with open(path, 'rb') as f:
            img = ExifImage(f)
            if img.has_exif:
                if hasattr(img, 'datetime_original'): raw_date = img.datetime_original
                if hasattr(img, 'gps_latitude'):
                    lat = get_decimal_from_dms(img.gps_latitude, img.gps_latitude_ref)
                    lon = get_decimal_from_dms(img.gps_longitude, img.gps_longitude_ref)
    except Exception: pass
    if lat is None: lat, lon = get_gps_from_pillow(path)
    return {
        "date": format_date_str(raw_date),
        "place": get_geocoded_address(lat, lon) if lat is not None else "No GPS Data"
    }

def overlay_text(img, text_lines):
    h, w = img.shape[:2]
    # Font settings optimized for 1080p
    font_scale = 1.2
    thickness = 2
    margin_x, margin_y = 50, 60
    line_spacing = 50
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = h - margin_y
    for line in reversed([l for l in text_lines if l]):
        # Shadow/Outline
        cv2.putText(img, str(line), (margin_x + 2, y + 2), font, font_scale, (0,0,0), thickness + 2, cv2.LINE_AA)
        # Main text
        cv2.putText(img, str(line), (margin_x, y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
        y -= line_spacing

def resize_and_pad_pil(pil_img, target_w=1920, target_h=1080):
    # Use PIL for images as it handles color profiles and high-quality downsampling better
    w, h = pil_img.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # PIL.Image.Resampling.LANCZOS is excellent for photos
    resized = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
    
    canvas = PILImage.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

def resize_and_pad(img, target_w=1920, target_h=1080):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # For video frames, INTER_AREA is superior for downscaling to prevent moire/shimmering
    # For upscaling, INTER_CUBIC is preferred over Lanczos for video to avoid ringing
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

class Slideshow:
    def __init__(self, files):
        self.files = files
        self.index = 0
        self.paused = False
        self.fullscreen = False
        self.window_name = "Shuffle Slideshow"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        self.next_item = False
        self.prev_item = False

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        prop = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, prop)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.toggle_fullscreen()
        elif event == cv2.EVENT_LBUTTONDOWN:
            rect = cv2.getWindowImageRect(self.window_name)
            win_w = rect[2] if rect and rect[2] > 0 else 800
            if x < win_w // 2: self.prev_item = True
            else: self.next_item = True
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.paused = not self.paused
        elif event == cv2.EVENT_RBUTTONDOWN: 
            self.prev_item = True

    def handle_keys(self, key):
        if key == -1 or key == 255: return False
        key = key & 0xFF
        if key == 27: cleanup_and_exit()
        if key == 32: self.paused = not self.paused; return False
        if key == ord('f'): self.toggle_fullscreen(); return False
        # 83/81 = Right/Left arrow (X11 specific)
        if key in [ord('d'), ord('n'), 83]: self.next_item = True; return True
        if key in [ord('a'), ord('p'), 81]: self.prev_item = True; return True
        return False

    def play(self):
        while True:
            self.next_item = False
            self.prev_item = False
            file_path = self.files[self.index]
            print(f"[{self.index + 1}/{len(self.files)}] {os.path.basename(file_path)}")
            if file_path.lower().endswith(('.jpg', '.jpeg')): self.show_image(file_path)
            else: self.show_video(file_path)
            self.index = (self.index + (-1 if self.prev_item else 1)) % len(self.files)

    def show_image(self, path):
        try:
            with PILImage.open(path) as pil_img:
                # Apply EXIF Orientation flag
                pil_img = ImageOps.exif_transpose(pil_img)
                # High quality resizing to 1080p using PIL
                img = resize_and_pad_pil(pil_img)
        except Exception: return

        meta = get_image_metadata(path)
        overlay_text(img, [meta["date"], meta["place"]])
        
        start_time = time.time()
        while True:
            cv2.imshow(self.window_name, img)
            key = cv2.waitKey(100)
            if self.handle_keys(key) or self.next_item or self.prev_item: break
            if not self.paused:
                if (time.time() - start_time) * 1000 >= TIMER_DELAY: break
            else: start_time = time.time()

    def show_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error opening video: {path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time_ms = 1000.0 / fps if fps > 0 else 33.3
        
        mod_date = "Unknown Date"
        try:
            mod_date = format_date_str(datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d'))
        except (OSError, ValueError): pass
        
        last_frame_with_overlay = None
        prev_timestamp = 0
        
        while cap.isOpened():
            if not self.paused:
                t_start = time.time()
                ret, frame = cap.read()
                if not ret: break
                
                frame = resize_and_pad(frame)
                overlay_text(frame, [mod_date, "Video"])
                cv2.imshow(self.window_name, frame)
                last_frame_with_overlay = frame
                
                # Use video timestamp if available, otherwise fall back to FPS
                curr_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                if curr_timestamp > 0 and prev_timestamp > 0:
                    wait_ms = max(1, int(curr_timestamp - prev_timestamp))
                else:
                    elapsed_ms = (time.time() - t_start) * 1000
                    wait_ms = max(1, int(frame_time_ms - elapsed_ms))
                prev_timestamp = curr_timestamp
            else:
                if last_frame_with_overlay is not None:
                    cv2.imshow(self.window_name, last_frame_with_overlay)
                wait_ms = 30
            
            key = cv2.waitKey(wait_ms)
            if self.handle_keys(key) or self.next_item or self.prev_item: break
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default=".")
    args = parser.parse_args()
    search_dir = os.path.expanduser(args.directory)
    all_files = [os.path.join(search_dir, f) for f in os.listdir(search_dir) if f.lower().endswith(('.jpg', '.jpeg', '.mp4'))]
    
    files = []
    seen = set()
    for f in all_files:
        resolved = os.path.realpath(f)
        if os.path.exists(resolved) and resolved not in seen:
            seen.add(resolved)
            files.append(resolved)

    if not files: print("No files found.")
    else:
        random.shuffle(files)
        Slideshow(files).play()
