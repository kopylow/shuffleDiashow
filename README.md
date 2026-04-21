# Shuffle Slideshow Tool

A command-line Python tool that opens a window to show a shuffled slideshow of JPG and MP4 files, displaying their date and location (City/Country) from metadata.

## Features
- **Shuffle:** Plays files in a random order.
- **Metadata:** Extracts date and GPS coordinates from images.
- **Geocoding:** Converts GPS coordinates to City/Country using OpenStreetMap (Nominatim).
- **Controls:**
  - **Mouse:** 
    - Click **Left Side**: Previous item.
    - Click **Right Side**: Next item.
    - Click **Center**: Play/Pause.
  - **Keyboard:**
    - `Space`: Play/Pause.
    - `D` or `Right Arrow`: Next.
    - `A` or `Left Arrow`: Previous.
    - `ESC`: Exit.
- **Videos:** Automatically waits for the video to finish before moving to the next item (no sound).

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) System dependencies:**
   Depending on your OS, you might need extra libraries for OpenCV (video handling). For Ubuntu/Debian:
   ```bash
   sudo apt-get install libopencv-dev
   ```

## Usage

Run the script from the directory containing your media, or provide a path:

```bash
python shuffle_slideshow.py /path/to/your/media
```

## Configuration
You can adjust the `TIMER_DELAY` (default 5000ms) in `shuffle_slideshow.py` to change how long images are displayed.
