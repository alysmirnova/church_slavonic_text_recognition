import cv2
import numpy as np
from heapq import *
from segmentation.thresholding import thresholding


def get_peak_regions(hpp):
    divider = 1.8
    threshold = (np.max(hpp)-np.min(hpp))/divider
    peaks = []
    for i, hppv in enumerate(hpp):
        if hppv < threshold:
            peaks.append([i, hppv])
    return peaks


def get_walking_regions_between_lines(peaks):
    hpp_clusters = []
    cluster = []
    for index, value in enumerate(peaks):
        cluster.append(value)
        if index < len(peaks)-1 and peaks[index+1] - value > 1:
            hpp_clusters.append(cluster)
            cluster = []
        if index == len(peaks)-1:
            hpp_clusters.append(cluster)
            cluster = []
    return hpp_clusters


def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2


def astar(array, start, goal):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heappush(oheap, (fscore[start], start))
    while oheap:
        current = heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
    return []


def existence_of_path(window_image):
    if 0 in np.sum(window_image, axis=1):
        return True
    padded_window = np.zeros((window_image.shape[0], 1))
    world_map = np.hstack((padded_window, np.hstack((window_image, padded_window))))
    path = np.array(astar(world_map, (int(world_map.shape[0]/2), 0), (int(world_map.shape[0]/2), world_map.shape[1])))
    if len(path) > 0:
        return True
    return False


def get_road_block_regions(nmap):
    road_blocks = []
    needtobreak = False
    for col in range(nmap.shape[1]):
        start = col
        end = col+20
        if end > nmap.shape[1]-1:
            end = nmap.shape[1]-1
            needtobreak = True
        if not existence_of_path(nmap[:, start:end]):
            road_blocks.append(col)
        if needtobreak:
            break
    return road_blocks


def merge_road_blocks(road_blocks):
    blocks_cluster_groups = []
    blocks_cluster = []
    size = len(road_blocks)
    for index, value in enumerate(road_blocks):
        blocks_cluster.append(value)
        if index < size-1 and (road_blocks[index+1] - road_blocks[index]) > 1:
            blocks_cluster_groups.append([blocks_cluster[0], blocks_cluster[len(blocks_cluster)-1]])
            blocks_cluster = []
        if index == size-1 and len(blocks_cluster) > 0:
            blocks_cluster_groups.append([blocks_cluster[0], blocks_cluster[len(blocks_cluster)-1]])
            blocks_cluster = []
    return blocks_cluster_groups


def cut_line_from_image(image, lower_line, upper_line):
    lower_boundary = np.min(lower_line[:, 0])
    upper_boundary = np.max(upper_line[:, 0])
    image_copy = np.copy(image)
    r, c = image_copy.shape
    for index in range(c-1):
        image_copy[0:lower_line[index, 0], index] = 255
        image_copy[upper_line[index, 0]:r, index] = 255
    return image_copy[lower_boundary:upper_boundary, :]


def split_indexes_into_groups(peaks):
    len_peaks = [[peaks[0]]]
    for i in range(1, len(peaks)):
        if peaks[i] == peaks[i-1] + 1:
            len_peaks[len(len_peaks)-1].append(peaks[i])
        else:
            len_peaks.append([peaks[i]])
    return len_peaks


def add_short_lines(peaks_index, hpp):
    hpp_peaks = split_indexes_into_groups(peaks_index)
    avg_len = sum([len(x) for x in hpp_peaks]) / len(hpp_peaks)
    for i in range(len(hpp_peaks)):
        if len(hpp_peaks[i]) <= avg_len / 2:
            left_hpp = hpp_peaks[i - 1][-1] if i != 0 else -1000
            right_hpp = hpp_peaks[i + 1][0] if i != len(hpp_peaks) - 1 else 1000
            if (hpp_peaks[i][0] - left_hpp) < (right_hpp - hpp_peaks[i][-1]):
                for j in range(left_hpp + 1, hpp_peaks[i][0]):
                    peaks_index = np.append(j, peaks_index)
            else:
                for j in range(hpp_peaks[i][-1] + 1, right_hpp):
                    peaks_index = np.append(j, peaks_index)
    peaks_index.sort()
    divider = 1.8
    threshold = (np.max(hpp) - np.min(hpp)) / divider
    inv_peaks_index = []
    for i in range(len(hpp)):
        if i not in peaks_index:
            inv_peaks_index.append(i)
    inv_hpp_peaks = split_indexes_into_groups(inv_peaks_index)
    length = [len(x) for x in inv_hpp_peaks]
    max_len = max(length)
    min_len = max(length) if min(length) <= max(length)/2 else min(length)
    min_threshold = np.min(hpp) + 1000
    h = (threshold - min_threshold) / 10
    new_peaks = []
    while min_threshold < threshold:
        peaks = []
        for i, hppv in enumerate(hpp):
            if hppv > min_threshold:
                peaks.append(i)
        len_peaks = split_indexes_into_groups(peaks)
        peaks = []
        for lenp in len_peaks:
            if min_len <= len(lenp) <= max_len:
                for le in lenp:
                    peaks.append(le)
        peaks = [x for x in peaks if x not in inv_peaks_index]
        if len(peaks):
            len_peaks = split_indexes_into_groups(peaks)
            for lenp in len_peaks:
                if min_len <= len(lenp) <= max_len:
                    for le in lenp:
                        if le not in new_peaks:
                            new_peaks.append(le)
        min_threshold += h
    for npeak in new_peaks:
        peaks_index = peaks_index[peaks_index != npeak]
    return peaks_index


def line_segmentation(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobel_img = np.sqrt(gx * gx + gy * gy)
    hpp = np.sum(sobel_img, axis=1)
    peaks = get_peak_regions(hpp)
    peaks_index = np.array(peaks)[:, 0].astype(int)
    peaks_index = add_short_lines(peaks_index, hpp)

    segmented_img = np.copy(img)
    h, w = segmented_img.shape
    for i in range(h):
        if i in peaks_index:
            segmented_img[i, :] = 0
    hpp_clusters = get_walking_regions_between_lines(peaks_index)
    binary_image = thresholding(img)

    for cluster_of_interest in hpp_clusters:
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest) - 1], :]
        road_blocks = get_road_block_regions(nmap)
        road_blocks_cluster_groups = merge_road_blocks(road_blocks)
        val = 10
        for index, road_blocks in enumerate(road_blocks_cluster_groups):
            window_image = nmap[:, road_blocks[0]: road_blocks[1] + val]
            window_image *= 0
            binary_image[cluster_of_interest[0]:cluster_of_interest[-1], :][:,
                road_blocks[0]: road_blocks[1] + val] = window_image
    line_segments = []
    for i, cluster_of_interest in enumerate(hpp_clusters):
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest) - 1], :]
        path = np.array(astar(nmap, (int(nmap.shape[0] / 2), 0), (int(nmap.shape[0] / 2), nmap.shape[1] - 1)))
        if len(path) > 0:
            offset_from_top = cluster_of_interest[0]
            path[:, 0] += offset_from_top
            line_segments.append(path)

    last_bottom_row = np.flip(np.column_stack(((np.ones((img.shape[1],)) * img.shape[0]),
                                               np.arange(img.shape[1]))).astype(int), axis=0)
    line_segments.append(last_bottom_row)

    line_images = []
    line_count = len(line_segments)

    for line_index in range(line_count - 1):
        line_image = cut_line_from_image(img, line_segments[line_index], line_segments[line_index + 1])
        line_images.append(line_image)

    sum_of_string_lengths = 0
    for line_index in range(line_count - 1):
        sum_of_string_lengths += np.min(line_segments[line_index + 1][:, 0]) - \
                                 np.min(line_segments[line_index][:, 0])
    avg = sum_of_string_lengths / line_count

    for line_index in range(line_count - 1):
        line_width = np.min(line_segments[line_index + 1][:, 0]) - np.min(line_segments[line_index][:, 0])
        if line_width < avg // 2 and line_index == line_count - 2:
            line_images.pop(line_index)
    return line_images
