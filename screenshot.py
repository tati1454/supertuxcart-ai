import base64
from io import BytesIO

from Xlib import X, display
from Xlib.xobject import drawable

from PIL import Image


def take_screenshot(window: drawable.Window, convert_to_base64=False):
    size = window.get_geometry()
    raw = window.get_image(0, 0, size.width, size.height, X.ZPixmap, 0xffffffff)
    image = Image.frombytes("RGB", (size.width, size.height), raw.data, "raw", "BGRX").convert("L").resize((256, 144))

    if convert_to_base64:
        stream = BytesIO()
        image.save(stream, format="JPEG")
        return base64.b64encode(stream.getvalue()).decode()
    else:
        return image

def get_window_by_name(name, root_window: drawable.Window=None):
    dplay = display.Display()
    if root_window is None:
        root_window = dplay.screen().root
    windows_tree: list[drawable.Window] = root_window.query_tree().children
    if name == "root":
        return root_window

    for wnd in windows_tree:
        if wnd.get_wm_name() == name:
            window_geometry = wnd.get_geometry()
            if window_geometry.width == 1 and window_geometry.height == 1:
                continue
            return wnd
        elif wnd.get_wm_name() is None:
            rwnd = get_window_by_name(name, root_window=wnd)
            if rwnd is not None:
                return rwnd
