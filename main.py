import os
import shutil
import sys

from Gui.guiwindow import GuiWindow


# pyinstaller --onefile --windowed --add-data "venv\Lib\site-packages\facenet_pytorch\data\;facenet_pytorch\data" main.py

def main():
    torch_home = os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
        )
    )
    __DIR__ = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    src_path = os.path.join(__DIR__, 'facenet_pytorch', 'data', '20180402-114759-vggface2.pt')
    dest_path = os.path.join(torch_home, 'checkpoints', '20180402-114759-vggface2.pt')
    if not os.path.exists(torch_home):
        os.mkdir(torch_home)
        if not os.path.exists(os.path.join(torch_home, 'checkpoints')):
            os.mkdir(os.path.join(torch_home, 'checkpoints'))
    if not os.path.exists(dest_path):
        shutil.copy(src_path, dest_path)
    window = GuiWindow()
    window.root.mainloop()


if __name__ == '__main__':
    main()
