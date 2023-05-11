from __future__ import annotations

import collections
import dataclasses
import glob
import os
import typing

import numpy as np
import pandas as pd
import cv2


CSV_FILE_INPUT = "2023-05/out/김동주.csv"
CSV_FILE_OUTPUT = "2023-05/out/bbox.csv"

DATA_DIR = '/Users/hepheir/GitHub/smu-cclab/WSPDUS/data/Products10k/train'


KEY_BACKSPACE = 127
KEY_SPACE = 32
KEY_ENTER = 13
KEY_ESC = 27
KEY_A = ord('a')
KEY_D = ord('d')
KEY_E = ord('e')
KEY_Q = ord('q')
KEY_Z = ord('z')
KEY_X = ord('x')


TYPE_NO_PROB = 0
TYPE_CHECK_LATER = 1
BBOX_NOT_DECIDED = -1

COLORS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
}

class FileNameLoader:
    names: typing.Deque[str]

    def __init__(self, dir: str) -> None:
        self.dir = dir
        self.names = collections.deque()
        self._load_filenames_via_csv(CSV_FILE_INPUT)

    def __str__(self) -> str:
        return f'<img: {self.get_path()}>'

    def __repr__(self) -> str:
        return self.__str__()

    def _load_via_glob(self):
        names = glob.glob('*.'+self.ext, root_dir=self.dir)
        names.sort(key=self._filename_indexing)
        self.names.extend(names)

    def _load_filenames_via_csv(self, csv_file: str):
        df = pd.read_csv(csv_file)
        names = list(df['name'].values)
        names.sort(key=self._filename_indexing)
        self.names.extend(names)

    def _filenames(self) -> typing.List[str]:
        return glob.glob('*.'+self.ext, root_dir=self.dir)

    def _filename_indexing(self, filename: str) -> int:
        return int(os.path.splitext(filename)[0])

    def _filename_abspath(self, filename: str) -> str:
        return os.path.join(self.dir, filename)

    def prev(self) -> None:
        self.names.rotate(1)
        print(self)

    def next(self) -> None:
        self.names.rotate(-1)
        print(self)

    def get_path(self) -> str:
        return self.names[0]

    def get_full_path(self) -> str:
        return self._filename_abspath(self.get_path())


@dataclasses.dataclass
class Coordinates:
    x: int
    y: int


@dataclasses.dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def of(cls, c1: Coordinates, c2: Coordinates) -> BoundingBox:
        x = min(c1.x, c2.x)
        y = min(c1.y, c2.y)
        w = abs(c1.x - c2.x)
        h = abs(c1.y - c2.y)
        return BoundingBox(x, y, w, h)

    def __iter__(self) -> typing.Iterator[int]:
        yield self.x
        yield self.y
        yield self.w
        yield self.h


class Record:
    @classmethod
    def get_columns(cls) -> typing.List[str]:
        return ['name', 'x', 'y', 'w', 'h', 'type']

    name: str
    x: int
    y: int
    w: int
    h: int
    type: int

    def __init__(self,
                 name: str,
                 x: int = BBOX_NOT_DECIDED,
                 y: int = BBOX_NOT_DECIDED,
                 w: int = BBOX_NOT_DECIDED,
                 h: int = BBOX_NOT_DECIDED,
                 type: int = TYPE_NO_PROB) -> None:
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.type = type

    def __iter__(self) -> typing.Iterator[int | str]:
        yield self.name
        yield self.x
        yield self.y
        yield self.w
        yield self.h
        yield self.type

    def is_saved(self) -> bool:
        return (self.x != BBOX_NOT_DECIDED) and (self.y != BBOX_NOT_DECIDED) and (self.w != BBOX_NOT_DECIDED) and (self.h != BBOX_NOT_DECIDED)



class RecordRepository:
    def __init__(self, csv_file: str) -> None:
        self.csv_file = csv_file
        self.rows: typing.Dict[str, Record] = {}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.csv_file):
            df = pd.read_csv(self.csv_file)
            for i in range(len(df)):
                rec = Record(*df.loc[i].values)
                self.rows[rec.name] = rec

    def save(self) -> None:
        data = list(map(lambda x: list(iter(x)), self.rows.values()))
        df = pd.DataFrame(data, columns=Record.get_columns())
        df.to_csv(self.csv_file, index=False)

    def get(self, name: str) -> Record:
        if name not in self.rows:
            self.rows[name] = Record(name)
        return self.rows[name]


class WindowHandler:
    def __init__(self, winname: str) -> None:
        self.bg: np.ndarray = np.zeros((100, 100))
        self.out: np.ndarray = self.bg
        self.winname: str = winname
        cv2.namedWindow(self.winname, cv2.WINDOW_AUTOSIZE)

    def run(self, strict_mode: bool = False) -> None:
        self.render()
        if strict_mode:
            self.loop()
        else:
            try:
                self.loop()
            except Exception as e:
                print(e)
                self.on_error()

    def loop(self) -> None:
        while True:
            if (key := cv2.waitKeyEx(10)) != -1:
                self.on_key_press(key)


    def set_background(self, image: np.ndarray) -> None:
        self.bg = image
        self.render()

    def render(self) -> None:
        cv2.imshow(self.winname, self.out)
        cv2.setMouseCallback(self.winname, self.mouse_event, self.out)

    def mouse_event(self, event:int, x: int, y: int, *args) -> None:
        coordinates = Coordinates(x, y)
        if event == cv2.EVENT_MOUSEMOVE:
            self.on_mouse_move(coordinates)
        if event == cv2.EVENT_FLAG_LBUTTON:
            self.on_mouse_click(coordinates)

    def on_mouse_move(self, coordinates: Coordinates) -> None:
        self.render()

    def on_mouse_click(self, coordinates: Coordinates) -> None:
        self.render()

    def on_key_press(self, key: int) -> None:
        self.render()

    def on_error(self) -> None:
        pass


class Labeler(WindowHandler):
    def __init__(self, data_dir: str, csv_file: str, winname: str = 'image labeling') -> None:
        super().__init__(winname)
        self.mouse: Coordinates = Coordinates(0, 0)
        self.savedMouse: typing.Optional[Coordinates] = None
        self.fileNameLoader = FileNameLoader(data_dir)
        self.recordRepository = RecordRepository(csv_file)

    @property
    def record(self) -> Record:
        return self.recordRepository.get(self.fileNameLoader.get_path())

    def run(self, strict_mode: bool = False) -> None:
        self.load_image()
        return super().run(strict_mode)

    def render(self) -> None:
        self.out = self.bg.copy()
        self.draw_saved_bbox()
        self.draw_ongoing_bbox()
        self.draw_cursor()
        return super().render()

    def draw_cursor(self) -> None:
        color = (0, 0, 255)
        thickness = 1
        h, w = self.out.shape[:2]
        cv2.line(self.out, (self.mouse.x, 0), (self.mouse.x, h), color, thickness)
        cv2.line(self.out, (0, self.mouse.y), (w, self.mouse.y), color, thickness)

    def draw_saved_bbox(self) -> None:
        if self.record.is_saved():
            point1 = (self.record.x, self.record.y)
            point2 = (self.record.x+self.record.w, self.record.y+self.record.h)
            color = COLORS['green'] if self.record.type == TYPE_NO_PROB else COLORS['yellow']
            cv2.rectangle(self.out, point1, point2, color=color, thickness=1)

    def draw_ongoing_bbox(self) -> None:
        if self.savedMouse is not None:
            point1 = (self.savedMouse.x, self.savedMouse.y)
            point2 = (self.mouse.x, self.mouse.y)
            cv2.rectangle(self.out, point1, point2, color=COLORS['red'], thickness=1)

    def on_key_press(self, key: int) -> None:
        if key == KEY_SPACE:
            self.save_mouse_coordinates()
        if key == KEY_Q:
            self.drop_mouse_coordinates()
        if key == KEY_A:
            self.load_prev_image()
        if key == KEY_D:
            self.load_next_image()
        if key == KEY_E:
            self.toggle_image_type()
        if key == KEY_Z:
            self.mouse.x = 0
            self.mouse.y = 0
            self.save_mouse_coordinates()
        if key == KEY_X:
            self.mouse.x = self.bg.shape[1]-1
            self.mouse.y = self.bg.shape[0]-1
            self.save_mouse_coordinates()
        if key == KEY_ESC:
            raise StopIteration()
        self.recordRepository.save()
        super().on_key_press(key)

    def on_mouse_move(self, coordinates: Coordinates) -> None:
        self.mouse = coordinates
        return super().on_mouse_move(coordinates)

    def save_mouse_coordinates(self) -> None:
        if self.savedMouse is None:
            self.savedMouse = self.mouse
        else:
            self.record.x = min(self.mouse.x, self.savedMouse.x)
            self.record.y = min(self.mouse.y, self.savedMouse.y)
            self.record.w = abs(self.mouse.x - self.savedMouse.x)
            self.record.h = abs(self.mouse.y - self.savedMouse.y)
            self.drop_mouse_coordinates()

    def drop_mouse_coordinates(self) -> None:
        self.savedMouse = None

    def toggle_image_type(self) -> None:
        if self.record.type == TYPE_NO_PROB:
            self.record.type = TYPE_CHECK_LATER
        else:
            self.record.type = TYPE_NO_PROB

    def load_prev_image(self) -> None:
        self.drop_mouse_coordinates()
        self.fileNameLoader.prev()
        self.load_image()

    def load_next_image(self) -> None:
        self.drop_mouse_coordinates()
        self.fileNameLoader.next()
        self.load_image()

    def load_image(self) -> None:
        self.set_background(cv2.imread(self.fileNameLoader.get_full_path()))


if __name__ == '__main__':
    Labeler(DATA_DIR, CSV_FILE_OUTPUT).run()
