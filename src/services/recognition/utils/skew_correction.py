from collections.abc import Sequence

import cv2
import numpy as np

from .shapes import Line, Point, Rectangle, find_intersection, get_distance


def correct_skew(
        src: np.ndarray,
        ref: np.ndarray,
        max_features: int = 2000,
        good_percent: float = 0.05,
) -> np.ndarray:
    # Image Features & Alignment with OpenCV
    # https://www.youtube.com/watch?v=_o6fSMCmNnQ

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Detect ORB Features and compute descriptors
    orb = cv2.ORB.create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(ref_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(src_gray, None)

    # Match features
    matcher = cv2.DescriptorMatcher.create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)

    # Remove not good features
    good_count = min(int(len(matches) * good_percent), 50)
    matches = matches[:good_count]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography to wrap image
    height, width, channels = src.shape
    return cv2.warpPerspective(src, h, (width, height))


def detect_corners(  # noqa: C901, PLR0912
        src: np.ndarray,
        *,
        roi_centerline: Rectangle,
        roi_bandwidth: int,
        edge_min_length: int,
    ) -> tuple[list[Point | None], np.ndarray | None]:

    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.GaussianBlur(src_gray, (5, 5), 0)

    output = cv2.cvtColor(src_gray.copy(), cv2.COLOR_GRAY2BGR)

    lsd = cv2.createLineSegmentDetector(0)
    segments = lsd.detect(src_gray)[0]

    inner = Rectangle(
        roi_centerline.x + roi_bandwidth / 2,
        roi_centerline.y + roi_bandwidth / 2,
        roi_centerline.width - roi_bandwidth,
        roi_centerline.height - roi_bandwidth,
    )
    cv2.rectangle(output, inner.p1.tup, inner.p3.tup, (0, 255, 0), 2)

    outer = Rectangle(
        roi_centerline.x - roi_bandwidth / 2,
        roi_centerline.y - roi_bandwidth / 2,
        roi_centerline.width + roi_bandwidth,
        roi_centerline.height + roi_bandwidth,
    )
    cv2.rectangle(output, outer.p1.tup, outer.p3.tup, (0, 255, 0), 2)

    edges = []
    for segment in segments:
        line = Line(*segment[0])
        if (line.length > edge_min_length and
                    outer.contains_point(line.p1) and
                    outer.contains_point(line.p2) and
                    not inner.contains_point(line.p1) and
                    not inner.contains_point(line.p2)):
            edges.append(line)
            cv2.line(output, line.p1.tup, line.p2.tup, (0, 255, 255), 2)

    intersects = []
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            intersect = find_intersection(edges[i], edges[j])
            if intersect is None:
                continue
            if outer.contains_point(intersect):
                d1 = get_distance(edges[i].p1, intersect)
                d2 = get_distance(edges[i].p2, intersect)
                if min(d1, d2) > edges[i].length:
                    continue
                d3 = get_distance(edges[j].p1, intersect)
                d4 = get_distance(edges[j].p2, intersect)
                if min(d3, d4) > edges[j].length:
                    continue
                intersects.append(intersect)

    frame_corners = ((inner.p1, inner.p3),
                     (inner.p2, inner.p4),
                     (inner.p3, inner.p1),
                     (inner.p4, inner.p2))

    detected = []
    for i in range(len(frame_corners)):
        max_distance = get_distance(frame_corners[i][0], frame_corners[i][1])
        corner = None
        for intersect in intersects:
            distance = get_distance(intersect, frame_corners[i][1])
            if distance > max_distance:
                max_distance = distance
                corner = intersect
        detected.append(corner)

    for corner in detected:
        if corner is not None:
            cv2.circle(output, corner.tup, 8, (0, 255, 255), -1)

    return detected, output


def adjust_corners(
        src: np.ndarray,
        corner_points: Sequence[Point],
        dest: Rectangle,
        *,
        margin: Sequence[int] | None = None,
) -> tuple[np.ndarray, list[Point]]:
    src_pts = np.float32([corner_points[0].tup, corner_points[1].tup,
                          corner_points[2].tup, corner_points[3].tup])

    if margin is None:
        margin = [0, 0, 0, 0]

    margin_top, margin_right, margin_bottom, margin_left = margin

    width = int(dest.width)
    height = int(dest.height)
    dest_width = margin_left + width + margin_right
    dest_height = margin_top + height + margin_bottom

    dest_pts = np.float32(
        [[margin_left, margin_top], [margin_left + width, margin_top],
         [margin_left + width, margin_top + height], [margin_left, margin_top + height]])

    matrix = cv2.getPerspectiveTransform(src_pts, dest_pts)
    dest = cv2.warpPerspective(src, matrix, (dest_width, dest_height))

    dest_pts = [Point(p[0], p[1]) for p in dest_pts]

    return dest, dest_pts
