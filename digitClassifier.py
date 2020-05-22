from bs4 import BeautifulSoup
import numpy as np
import csv
from matplotlib import pyplot as plt
import randomForest as rfc
import joblib as jl


class Digit:

    def __init__(self, ui, traces):
        self.ui = ui
        self.traces = traces


def outputs(input, classification, classes, name):
    with open(name+'.txt', 'w') as o:
        for i in range(len(classification)):
            o.write(input[i].ui + ', ' + classes[int(classification[i])] + '\n')


def getlengthOfStroke(stroke):
    nextPoint = np.concatenate((stroke[1:, :], stroke[-1, :].reshape((1, stroke.shape[1]))))
    return np.sum(np.sqrt(np.sum(np.square(np.subtract(stroke, nextPoint)), axis=1, keepdims=True)))


def line_length(strokes):
    total = 0
    for stroke in strokes:
        total += getlengthOfStroke(stroke)
    return total, total/len(strokes)


def combined_strokes(strokes):
    combined = strokes[0]
    for i in range(1, len(strokes)-1):
        combined = np.r_[combined, strokes[i]]
    return combined


def covariance_points(strokes):
    combined = combined_strokes(strokes)
    if combined.shape[0] == 1:
        return 0
    return np.cov(combined[:, 0], combined[:, 1])[0, 1]


def mean_x_y(strokes):
    combined = combined_strokes(strokes)
    return np.mean(combined[:, 0]), np.mean(combined[:, 1])


def slope(p1, p2):
    return (p2[1] - p1[1])/((p2[0] - p1[1])+1e-9)


def num_of_sharp_points(strokes):
    sharp_points = 0
    for stroke in strokes:
        sharp_points += 2
        a = []
        for i in range(len(stroke)-1):
            a.append(slope(stroke[i], stroke[i+1]))
        for i in range(1, len(a)-1):
            theta = a[i] - a[i+1]
            theta2 = a[i-1] - a[i]
            if theta == 0:
                continue
            delta = theta * theta2
            if (delta <= 0) and (theta2 != 0):
                sharp_points += 1
    total_points = (combined_strokes(strokes)).shape[0]
    return sharp_points, sharp_points/total_points


def orientation(p1, p2, p3):
    val = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])
    if val > 0:
        return 1
    elif val < 0:
        return 2
    else:
        return 0


def onSegment(p1, p2, p3):
    return max(p1[0], p3[0]) >= p2[0] >= min(p1[0], p3[0]) and \
           max(p1[1], p3[1]) >= p2[1] >= min(p1[1], p3[1])


def line_intersect(p1, p2, q1, q2):
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if (o1 != o2) and (o3 != o4):
        return True

    if (o1 == 0) and onSegment(p1, q1, p2):
        return True

    if (o2 == 0) and onSegment(p1, q2, p2):
        return True

    if (o3 == 0) and onSegment(q1, p1, q2):
        return True

    if (o4 == 0) and onSegment(q1, p2, q2):
        return True

    return False


def horizontal_vertical_lines(minV, maxV):
    ys = np.linspace(minV, maxV, 6)
    xs = np.array([minV, maxV])
    horizontal = []
    vertical = []
    for i in range(ys.shape[0]-1):
        gy = np.linspace(ys[i]+.05, ys[i+1]-.05, 9)
        x, y = np.meshgrid(xs, gy)
        horizontal.append([np.c_[x[:, 0], y[:, 0]], np.c_[x[:, 1], y[:, 1]]])
        vertical.append([np.c_[y[:, 0], x[:, 0]], np.c_[y[:, 1], x[:, 1]]])
    return np.array(horizontal),  np.array(vertical)


def plot_traces(stokes):
    for stroke in stokes:
        plt.plot(stroke[:, 0], stroke[:, 1])
    plt.ylim((-2, 2))
    plt.xlim((-2, 2))
    plt.show()


def crossing_features(strokes):
    h, v = horizontal_vertical_lines(-1, 1)
    h_total = []
    v_total = []
    # plot_traces(strokes)
    for hgroup, vgroup in zip(h[:], v[:]):
        htotal_cross = 0
        vtotal_cross = 0
        for i in range(9):
            hp1 = hgroup[0, i]
            hp2 = hgroup[1, i]
            vp1 = vgroup[0, i]
            vp2 = vgroup[1, i]
            for stroke in strokes:
                for j in range(len(stroke)-1):
                    if line_intersect(hp1, hp2, stroke[j], stroke[j+1]):
                        htotal_cross += 1
                    if line_intersect(vp1, vp2, stroke[j], stroke[j+1]):
                        vtotal_cross += 1
        h_total.append(htotal_cross/9)
        v_total.append(vtotal_cross/9)
    return h_total, v_total


def aspectRatio(coordinates):
    minX = []
    minY = []
    maxX = []
    maxY = []
    for stroke in coordinates:
        minX.append(np.min(stroke[:, 0]))
        minY.append(np.min(stroke[:, 1]))
        maxX.append(np.max(stroke[:, 0]))
        maxY.append(np.max(stroke[:, 1]))
    width = max(maxX) - min(minX)
    height = max(maxY) - min(minY)
    return height, width


def getNumberOfStroke(coordinates):
    return len(coordinates)


def fuzzyHist(coordinates):
    points = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    corners = np.column_stack((points[0].flatten(), points[1].flatten()))
    symbcoord = []
    for strokes in coordinates:
        for stroke in strokes:
            symbcoord.append(stroke)
    p1 = np.array(symbcoord)
    p2 = np.concatenate((p1[1:, :], p1[-1, :].reshape((1, p1.shape[1]))))
    horizVec = [2, 0]
    verticalVec = [0, -2]
    diag1Vec = [2, 2]
    diag2Vec = [2, -2]
    horizAxisLength = 2
    verticalAxisLength = 2
    diag1Length = 2.828
    diag2Length = 2.828
    bins = []
    for p3 in corners:
        cornerBins = []
        dist = np.linalg.norm(np.cross(p2[:-1] - p1[:-1], p1[:-1] - p3)) / ((np.linalg.norm(p2[:-1] - p1[:-1], axis=1)) + 1e-9)
        if dist.size == 0:
            cornerBins.append(np.array([0, 0, 0, 0]))
            bins.append(cornerBins)
            continue
        indices = dist.argsort()
        p1sorted = p1[indices]
        p2sorted = p2[indices]
        closestPoint1, closestPoint2 = p1sorted[:1, :], p2sorted[:1, :]
        lineVect = [closestPoint2[0][0] - closestPoint1[0][0], closestPoint2[0][1] - closestPoint1[0][1]]
        linelength = np.sqrt(np.square(lineVect[0]) + np.square(lineVect[1])) + 1e-9
        for vec, length in zip([horizVec, verticalVec, diag1Vec, diag2Vec], [horizAxisLength, verticalAxisLength, diag1Length, diag2Length]):
            if np.dot(lineVect, vec) / (linelength * length) < -1:
                ans = -1
            elif np.dot(lineVect, vec) / (linelength * length) > 1:
                ans = 1
            else:
                ans = np.dot(lineVect, vec) / (linelength * length)
            val = np.arccos(ans)
            cornerBins.append(val)
        bins.append(cornerBins)
    return np.array(bins).flatten()


def getdiagLength(strokes):
    point = []
    minX = []
    minY = []
    maxX = []
    maxY = []
    for stroke in strokes:
        minX.append(np.min(stroke[:, 0]))
        minY.append(np.min(stroke[:, 1]))
        maxX.append(np.max(stroke[:, 0]))
        maxY.append(np.max(stroke[:, 1]))
    point.append([min(minX), min(minY)])
    point.append([max(maxX), max(maxY)])
    return np.sqrt(np.sum(np.square(np.subtract(point[0], point[1]))))


def eliminateStrokes(data):
    for digit in data:
        strokes = digit.traces
        if len(strokes) > 1:
            newlist = []
            diagonalLength = getdiagLength(strokes)
            for stroke in strokes:
                length = getlengthOfStroke(stroke)
                if length > (0.1 * diagonalLength):
                    newlist.append(stroke)
            digit.traces = np.array(newlist)
            if digit.traces.shape[0] == 0:
                digit.traces = strokes
    return data


def features(data):
    for digit in data:
        trace = digit.traces
        digit.traces = normalize(trace)
    featureVector = []
    i = 1
    eliminated = eliminateStrokes(data)
    for digit in eliminated:
        perSampleFeature = []
        perSampleFeature.append(covariance_points(digit.traces))
        total, average = line_length(digit.traces)
        perSampleFeature.append(total)
        perSampleFeature.append(average)
        x, y = mean_x_y(digit.traces)
        perSampleFeature.append(x)
        perSampleFeature.append(y)
        total_sharp, average_sharp = num_of_sharp_points(digit.traces)
        perSampleFeature.append(total_sharp)
        perSampleFeature.append(average_sharp)
        height, width = aspectRatio(digit.traces)
        perSampleFeature.append(height)
        perSampleFeature.append(width)
        perSampleFeature.append(getNumberOfStroke(digit.traces))
        val1, val2 = crossing_features(digit.traces)
        for v1 in val1:
            perSampleFeature.append(v1)
        for v2 in val2:
            perSampleFeature.append(v2)
        hist = fuzzyHist(digit.traces)
        for h in hist:
            perSampleFeature.append(h)
        featureVector.append(perSampleFeature)
        if i % 1000 == 0:
            print(i)
        i += 1
    return featureVector


def classify_pipeline(featureVectors, model):
    return model.classify(np.array(featureVectors))


def classify_conf_pipeline(featureVectors, model):
    return model.classify(np.array(featureVectors))


def acc(pred, gt):
    print((1 - np.count_nonzero(np.subtract(pred, gt))/gt.shape[0]) * 100, "% accuracy")


def getAcc(pred, gt):
    acc = (1 - np.count_nonzero(np.subtract(pred, gt)) / gt.shape[0]) * 100
    print(acc, "% accuracy")
    return acc


def train(feat, classes, classifier):
    model = classifier(feat, classes)
    jl.dump(model, 'models/newRFCModel.txt', compress=5)
    return model


def fineTune(feat, classes, classifier, modelfold):
    model = classifier(feat, classes)
    jl.dump(model, 'models/fineTune/' + str(modelfold) + 'newRFCModel.txt', compress=5)
    return model


def fineTuneloadModel(modelFold):
    model = jl.load('models/fineTune/' + str(modelFold) + 'newRFCModel.txt')
    return model


def loadModel():
    model = jl.load('models/newRFCModel.txt')
    return model


def getFeatures(stroke):
    features(stroke)


def normalize(coordinates):
    if len(coordinates) > 1:
        minValX = []
        minValY = []
        maxValX = []
        maxValY = []
        newCoordinates = []
        for stroke in coordinates:
            minValX.append(np.min(stroke[:, 0]))
            minValY.append(np.min(stroke[:, 1]))
            maxValX.append(np.max(stroke[:, 0]))
            maxValY.append(np.max(stroke[:, 1]))
        for stroke in coordinates:
            if np.max(maxValX) - np.min(minValX) == 0 and np.max(maxValY) - np.min(minValY) == 0:
                newCoordinates.append(np.column_stack((np.zeros(stroke.shape[0]), np.zeros(stroke.shape[0]))))
            elif np.max(maxValX) - np.min(minValX) == 0:
                newCoordinates.append(
                    np.column_stack((np.zeros(stroke.shape[0]), ((2 * (stroke[:, 1] - np.min(minValY))) /
                                                                 (np.max(maxValY) - np.min(minValY))) - 1)))
            elif np.max(maxValY) - np.min(minValY) == 0:
                newCoordinates.append(np.column_stack((((2 * (stroke[:, 0] - np.min(minValX))) /
                                                        (np.max(maxValX) - np.min(minValX))) - 1,
                                                       np.zeros(stroke.shape[0]))))
            else:
                newCoordinates.append(np.column_stack((((2 * (stroke[:, 0] - np.min(minValX))) /
                                                        (np.max(maxValX) - np.min(minValX))) - 1,
                                                       ((2 * (stroke[:, 1] - np.min(minValY))) /
                                                        (np.max(maxValY) - np.min(minValY))) - 1)))
        return newCoordinates
    else:
        if np.max(coordinates[0][:, 0]) - np.min(coordinates[0][:, 0]) == 0 and np.max(coordinates[0][:, 1]) - np.min(
                coordinates[0][:, 1]) == 0:
            return [np.column_stack((np.zeros(coordinates[0].shape[0]), np.zeros(coordinates[0].shape[0])))]
        elif np.max(coordinates[0][:, 0]) - np.min(coordinates[0][:, 0]) == 0:
            return [np.column_stack(
                (np.zeros(coordinates[0].shape[0]), ((2 * (coordinates[0][:, 1] - np.min(coordinates[0][:, 1]))) /
                                                     (np.max(coordinates[0][:, 1]) - np.min(
                                                         coordinates[0][:, 1]))) - 1))]
        elif np.max(coordinates[0][:, 1]) - np.min(coordinates[0][:, 1]) == 0:
            return [np.column_stack((((2 * (coordinates[0][:, 0] - np.min(coordinates[0][:, 0]))) /
                                      (np.max(coordinates[0][:, 0]) - np.min(coordinates[0][:, 0]))) - 1,
                                     np.zeros(coordinates[0].shape[0])))]
        else:
            return [np.column_stack((((2 * (coordinates[0][:, 0] - np.min(coordinates[0][:, 0]))) /
                                      (np.max(coordinates[0][:, 0]) - np.min(coordinates[0][:, 0]))) - 1,
                                     ((2 * (coordinates[0][:, 1] - np.min(coordinates[0][:, 1]))) /
                                      (np.max(coordinates[0][:, 1]) - np.min(coordinates[0][:, 1]))) - 1))]


def createFakeData():
    n = np.random.randint(1, 5, size=1)[0]
    stroke = []
    for i in range(n):
        lengthOfStroke = np.random.randint(20, 30, size=1)[0]
        stroke.append(np.random.random((lengthOfStroke, 2)))
    return np.array(stroke)


if __name__ == '__main__':
    strokes = createFakeData()
    for stroke in strokes:
        normalize(stroke)
        getFeatures(stroke)
