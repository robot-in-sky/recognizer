import math


class Point:

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y

    @property
    def tup(self) -> tuple[int, int]:
        return int(self.x), int(self.y)

    def apply_scale(self, scale: float) -> None:
        self.x *= scale
        self.y *= scale


def get_distance(p1: Point, p2: Point) -> float:
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


class Line:

    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.p1 = Point(x1, y1)
        self.p2 = Point(x2, y2)

    @property
    def length(self) -> float:
        return get_distance(self.p1, self.p2)

    @property
    def k(self) -> float:
        return (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)

    @property
    def b(self) -> float:
        return self.p1.y - self.k * self.p1.x

    def apply_scale(self, scale: float) -> None:
        self.p1.apply_scale(scale)
        self.p2.apply_scale(scale)


class Rectangle:

    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        self.x, self.y = x, y
        self.width, self.height = width, height

    @property
    def p1(self) -> Point:
        return Point(self.x , self.y)

    @property
    def p2(self) -> Point:
        return Point(self.x + self.width, self.y)

    @property
    def p3(self) -> Point:
        return Point(self.x + self.width, self.y + self.height)

    @property
    def p4(self) -> Point:
        return Point(self.x, self.y + self.height)

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2, self.y + self.height / 2)

    def contains_point(self, p: Point) -> bool:
        return (self.x <= p.x <= self.x + self.width and
                self.y <= p.y <= self.y + self.height)

    def apply_scale(self, scale: float) -> None:
        self.x *= scale
        self.y *= scale
        self.width *= scale
        self.height *= scale


def find_intersection(l1: Line, l2: Line) -> Point | None:
    if l1.k == l2.k:
        return None
    x = (l2.b - l1.b) / (l1.k - l2.k)
    y = l1.k * x + l1.b
    return Point(x, y)
