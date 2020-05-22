import sys
from glob import iglob
from data_utility import *
import digitClassifier
import numpy as np
import os
import pickle as pk
import kmeans
import randomForest as rfc
from sklearn.cluster import KMeans


def read_in_data(file, path, ground_truth=False):
    """
    function to read in inkml data
    :param file: file containing a list of files to read in or all to be used when splitting
    :param path: path to upper level folder that contains all the training data
    :param ground_truth: boolean value determining if ground truth should be added
    :return: a list of Expressions
    """
    expersions = []
    if file == 'all':
        expersions = [inkml_to_Expression(path, f, ground_truth) for f in iglob(path+'/**/*.inkml', recursive=True)]
    elif os.path.splitext(file)[1][1:] == 'txt':
        with open(file) as o:
            expersions = [inkml_to_Expression(path, path+'/'+f.strip(), ground_truth) for f in o]
    elif os.path.splitext(file)[1][1:] == 'inkml':
        return [inkml_to_Expression(path, file, ground_truth)]
    return list(filter(lambda x: x is not None, expersions))


def split(path, outfile):
    expressions = read_in_data('all', path, True)

    counts, symbol_to_index = get_symbol_counts(expressions)

    train_counts = [0 for i in range(len(counts))]
    train = []
    test_counts = [0 for i in range(len(counts))]
    test = []

    def sort_key(elem):
        return min(list(map(lambda x: counts[symbol_to_index[x.classification]], elem.symbols)))
    sort_expressions = expressions.copy()
    sort_expressions.sort(key=sort_key)

    i = 0
    while len(sort_expressions) > 0:
        expression = sort_expressions.pop()
        if i % 3 != 0:
            train.append(expression)
            for symbol in expression.symbols:
                train_counts[symbol_to_index[symbol.classification]] += 1
        else:
            test.append(expression)
            for symbol in expression.symbols:
                test_counts[symbol_to_index[symbol.classification]] += 1
        i += 1

    probability_train = count_to_probability(train_counts)
    probability_test = count_to_probability(test_counts)

    print("Kl divergence test to train:", kl_divergance(probability_test, probability_train))

    with open(outfile + '_train_data.txt', 'w') as o:
        for expression in train:
            o.write(expression.file + '\n')
    with open(outfile + '_test_data.txt', 'w') as o:
        for expression in test:
            o.write(expression.file + '\n')


def objective_function(symbol_confidences):
    n = len(symbol_confidences)
    return np.power(np.prod(symbol_confidences), (1/n))


def strokes_mean(strokes):
    return np.array(list(map(lambda x: np.mean(x.points, 0), strokes)))


def time_order_segmentation(expression, recognition_model):
    strokes = expression.strokes.values()


def kmean_segementation(expression, get_clustering_features, recognition_model):
    strokes = expression.strokes.values()
    cluster_features = get_clustering_features(strokes)
    n = len(strokes)
    confidences = {}
    max_confidence = 0
    max_segmentation = []
    for k in range(n, n//4, -1):
        segmenter = kmeans.Kmeans(k)
        test_k = KMeans(k)
        test = test_k.fit_predict(cluster_features)
        # clusters = segmenter.kmeans(cluster_features)
        clusters = {}
        for i, index in enumerate(test):
            arr = clusters.get(index, None)
            if arr is None:
                clusters[index] = [i]
            else:
                arr.append(i)
        cluster_confidences = []
        cluster_symbols = []
        for key in clusters.keys():
            cluster = clusters[key]
            name = ''.join(list(map(str, cluster)))
            s, c = confidences.get(name, (None, None))
            if c is None:
                # needs to be changed for the max probability score
                symbol_strokes = []
                for stroke in cluster:
                    symbol_strokes.append(list(strokes)[stroke])
                s = Symbol(name, symbol_strokes)
                feature_vector = digitClassifier.features([s.to_digit()])
                c, classification = recognition_model.classify_conf(feature_vector)
                confidences[name] = (s, c)
                s.classification = recognition_model.int_to_class[classification] if classification < len(recognition_model.int_to_class) else 'junk'
                if s.classification == ',':
                    s.classification = 'COMMA'
                elif s.classification == 'junk':
                    c /= 10
                s.weight = c
            cluster_confidences.append(c)
            cluster_symbols.append(s)
        segmentation_confifdence = objective_function(cluster_confidences)
        if segmentation_confifdence > max_confidence:
            max_confidence = segmentation_confifdence
            max_segmentation = cluster_symbols
    expression.symbols = max_segmentation


def lg_or_output(expression, path):
    filepath = path + expression.file.replace('inkml', 'lg')
    dir = os.path.dirname(filepath)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(filepath, 'w') as o:
        o.write("# IUD, \"%s\"\n" % expression.uid)
        o.write("# Objects(%d):\n" % len(expression.symbols))
        for symbol in expression.symbols:
            name = ''
            strokes = ''
            for stroke in symbol.strokes:
                name += '%d:' % stroke.id
                strokes += ', %d' % stroke.id
            o.write('O, %s, %s, %.1f%s\n' % (name, symbol.classification, symbol.weight, strokes))


def test_output():
    test = Expression('test.inkml', 'testing-uid', [])
    test_s_1 = Symbol(0, [Stroke([], 1), Stroke([], 2), Stroke([], 3)])
    test_s_1.classification = 'A'
    test_s_2 = Symbol(1, [Stroke([], 0)])
    test_s_2.classification = '0'
    test.symbols += [test_s_1, test_s_2]
    lg_or_output(test, './')


def evaluate_for_all(file, path, outpath, segmentor):
    expressions = read_in_data(file, path)
    # exp = open('expressions.txt', 'wb')
    # pk.dump(expressions, exp)
    # exp.close()
    # exp = open('expressions.txt', 'rb')
    # expressions = pk.load(exp)
    recognition_model = digitClassifier.loadModel()
    for expression in expressions:
        # segment_and_classify(expression, recognition_model)
        # kmean_segementation(expression, strokes_mean, recognition_model)
        segmentor(expression, recognition_model)
        # segment_and_classify(expression, recognition_model)
        # kmean_segementation(expression, strokes_mean, recognition_model)
        lg_or_output(expression, outpath)


def baseline_segment(expression, recognition_model):
    expression.symbols = list(map(lambda x: Symbol(x.id, [x]), expression.strokes.values()))
    digits = list(map(lambda x: x.to_digit(), expression.symbols))
    featureVector = digitClassifier.features(digits)
    classifications = digitClassifier.classify_pipeline(featureVector, recognition_model)
    for symbol, classification in zip(expression.symbols, classifications):
        symbol.classification = recognition_model.int_to_class[classification] if classification < len(recognition_model.int_to_class) else 'junk'
        if symbol.classification == ',':
            symbol.classification = 'COMMA'
        symbol.weight = 1.0


def createStrokesandGT(data):
    digitData = []
    gTruth = []
    for exp in data:
        symbols = exp.symbols
        for symb in symbols:
            digitData.append(symb.to_digit())
            gTruth.append(symb.classification)
    return digitData, gTruth


def getLabels(gt, classes):
    labels = []
    for cl in gt:
        labels.append(classes.index(cl))
    return labels


def readTrainingJunk():
    trainJunkFeat = open('TrainJunkfeatureVec.txt', 'rb')
    train = pk.load(trainJunkFeat)
    junkTrainLabels = open('junkTrainLabels.txt', 'rb')
    labels = pk.load(junkTrainLabels)
    return train, labels


def readTestJunk():
    trainJunkFeat = open('TestJunkfeatureVec.txt', 'rb')
    train = pk.load(trainJunkFeat)
    junkTrainLabels = open('junkTestLabels.txt', 'rb')
    labels = pk.load(junkTrainLabels)
    return train, labels


def getClasses(gt):
    training_classes = {}
    training_int_classes = list(set(gt))
    for cl in training_int_classes:
        training_classes[cl] = training_int_classes.index(cl)
    return training_classes, training_int_classes


def createJunkStrokes(data):
    digits = []
    for exp in data:
        strokes = exp.strokes
        trace = []
        for stroke in strokes:
            trace.append(np.array(strokes[stroke].points).astype(np.float64))
        digits.append(digitClassifier.Digit(exp.uid, trace))
    return digits


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage [Python] [split/create/train/evaluate]')
        exit(1)
    if sys.argv[1] == 'split':
        if len(sys.argv) == 4:
            split(sys.argv[2], sys.argv[3])
        else:
            print('Usage [Python] split [path to data] [output file]')
    if sys.argv[1] == 'evaluate':
        if 5 <= len(sys.argv) <=6:
            seg_type = sys.argv[2]
            file = sys.argv[3]
            if len(sys.argv) == 5:
                path = ''
                outpath = sys.argv[4]
            else:
                path = sys.argv[4]
                outpath = sys.argv[5]
            seg = None
            if seg_type.lower() == 'baseline':
                seg = baseline_segment
            elif seg_type.lower() == 'kmean':
                def seg(expression, model): kmean_segementation(expression, strokes_mean, model)
            if seg is not None:
                evaluate_for_all(file, path, outpath, seg)
        else:
            print('Usage [Python] evaluate [segmentor] [Bulk file] [path to data] [path output]')
    if sys.argv[1] == 'create':
        if len(sys.argv) == 8:
            featureVectorpath = 'featureVec/'
            dataPath = 'data/'
            labelsAndclassesPath = 'labelsandClass/'
            groundTruthPath = 'groundTruth/'

            if not os.path.exists(featureVectorpath):
                os.makedirs(featureVectorpath)

            if not os.path.exists(dataPath):
                os.makedirs(dataPath)

            if not os.path.exists(labelsAndclassesPath):
                os.makedirs(labelsAndclassesPath)

            if not os.path.exists(groundTruthPath):
                os.makedirs(groundTruthPath)

            if not os.path.exists('models/'):
                os.makedirs('models/')

            ############### CREATE DATA ###################

            # reading train feature vectors
            data = read_in_data(sys.argv[2], sys.argv[6], True)
            dat = open(dataPath + 'data.txt', 'wb')
            pk.dump(data, dat)
            dat.close()
            digits, gt = createStrokesandGT(data)
            training_labels = getLabels(gt, list(set(gt)))
            training_classes, training_int_classes = getClasses(gt)

            train_featVect = digitClassifier.features(digits)
            trainfeat = open(featureVectorpath + 'trainingFeatVect.txt', 'wb')
            pk.dump(train_featVect, trainfeat)
            trainfeat.close()

            # reading train junk feature vectors
            junk = read_in_data(sys.argv[4], sys.argv[7], False)
            dat = open(dataPath + 'junk.txt', 'wb')
            pk.dump(junk, dat)
            dat.close()
            junkDigits = createJunkStrokes(junk)
            junkGt = ['junk' for i in range(len(junkDigits))]
            junk_labels = [len(training_int_classes) for i in range(len(junkDigits))]
            junk_classes, junk_int_classes = getClasses(junkGt)

            junk_featVect = digitClassifier.features(junkDigits)
            junkfeat = open(featureVectorpath + 'junkFeatVect.txt', 'wb')
            pk.dump(junk_featVect, junkfeat)
            junkfeat.close()

            ##################TEST DATA ######################

            # reading test feature vectors
            testdata = read_in_data(sys.argv[3], sys.argv[6], True)
            dat = open(dataPath + 'Testdata.txt', 'wb')
            pk.dump(testdata, dat)
            dat.close()
            testdigits, testgt = createStrokesandGT(testdata)

            test_featVect = digitClassifier.features(testdigits)
            testfeat = open(featureVectorpath + 'testingFeatVect.txt', 'wb')
            pk.dump(test_featVect, testfeat)
            testfeat.close()

            # reading junk test feature vectors
            junktest = read_in_data(sys.argv[5], sys.argv[7], False)
            dat = open(dataPath + 'junkTest.txt', 'wb')
            pk.dump(junktest, dat)
            dat.close()
            junkTestDigits = createJunkStrokes(junktest)
            junkTestGt = ['junk' for i in range(len(junkTestDigits))]

            junkTest_featVect = digitClassifier.features(junkTestDigits)
            junkTestfeat = open(featureVectorpath + 'junkTestFeatVect.txt', 'wb')
            pk.dump(junkTest_featVect, junkTestfeat)
            junkTestfeat.close()

            labels = np.r_[training_labels, junk_labels]
            classes = training_classes
            int_classes = training_int_classes
            groundtruth = (gt, junkGt, testgt, junkTestGt)

            label = open(labelsAndclassesPath + 'labels.txt', 'wb')
            pk.dump(labels, label)
            label.close()

            clas = open(labelsAndclassesPath + 'classes.txt', 'wb')
            pk.dump(classes, clas)
            clas.close()

            intClass = open(labelsAndclassesPath + 'intclasses.txt', 'wb')
            pk.dump(int_classes, intClass)
            intClass.close()

            groundT = open(groundTruthPath + 'groundTruth.txt', 'wb')
            pk.dump(groundtruth, groundT)
            groundT.close()
        else:
            print('Usage [Python] create [train_file] [junk_file] [test file] [junk test file] [path to data] [path to junk]')
    if sys.argv[1] == 'train':
        if len(sys.argv) == 3:
            config = sys.argv[2]
            if config == '0':
                print("Training using junk data")
            else:
                print("Training without junk data")
            featureVectorpath = 'featureVec/'
            dataPath = 'data/'
            labelsAndclassesPath = 'labelsandClass/'
            groundTruthPath = 'groundTruth/'

            trainfeat = open(featureVectorpath + 'trainingFeatVect.txt', 'rb')
            junkfeat = open(featureVectorpath + 'junkFeatVect.txt', 'rb')

            label = open(labelsAndclassesPath + 'labels.txt', 'rb')
            labels = pk.load(label)

            cls = open(labelsAndclassesPath + 'classes.txt', 'rb')
            classes = pk.load(cls)

            intClass = open(labelsAndclassesPath + 'intclasses.txt', 'rb')
            int_Classes = pk.load(intClass)

            groundT = open(groundTruthPath + 'groundTruth.txt', 'rb')
            groundtruth = pk.load(groundT)

            training_classes = classes
            training_int_classes = int_Classes
            gtTuple = groundtruth

            feat = np.array(pk.load(trainfeat))

            newlabels = labels[:feat.shape[0]]

            if config == '0':
                feat = np.r_[np.array(feat), np.array(pk.load(junkfeat))]
                newlabels = labels

            num_trees = 50
            depth = 20
            digitClassifier.train(feat, newlabels,
                                  lambda x1, x2: rfc.RFC(num_trees, depth, training_classes, training_int_classes, x1, x2))
            recognition_model = digitClassifier.loadModel()
            pred = digitClassifier.classify_pipeline(feat, recognition_model)
            digitClassifier.acc(pred, newlabels)

            testfeat = open(featureVectorpath + 'testingFeatVect.txt', 'rb')
            junkTestfeat = open(featureVectorpath + 'junkTestFeatVect.txt', 'rb')

            tFeat = pk.load(testfeat)
            jTestFeat = pk.load(junkTestfeat)

            testfeat = np.array(tFeat)
            if config == '0':
                testfeat = np.r_[tFeat, jTestFeat]

            test_labels = [recognition_model.class_to_int[c] for c in gtTuple[2]]
            junk_Testlabels = [len(recognition_model.int_to_class) for i in range(len(jTestFeat))]

            tLabels = np.array(test_labels)
            if config == '0':
                tLabels = np.r_[test_labels, junk_Testlabels]
            testpred = digitClassifier.classify_pipeline(testfeat, recognition_model)
            digitClassifier.acc(testpred, tLabels)

            parameters = [num_trees, depth]
            tuneTrees = np.tile(np.arange(parameters[0], 600, 100), 3)
            tuneDepth = np.arange(parameters[1], 35, 5).repeat(6)

            trainAccuracy = []
            testAccuracy = []
            folderNumber = 1
            print("Fine tuning model")
            for tr, de in zip(tuneTrees, tuneDepth):
                print("Model " + str(folderNumber) + "--> Parameters: " + str(tr) + " trees " + str(de) + " depth")
                print("\nTraining model . . .")
                digitClassifier.fineTune(feat, newlabels, lambda x1, x2: rfc.RFC(tr, de, training_classes, training_int_classes, x1, x2), folderNumber)
                recognition_model = digitClassifier.fineTuneloadModel(folderNumber)
                pred = digitClassifier.classify_pipeline(feat, recognition_model)
                print("\nTraining Accuracy ->", end="")
                trainAccuracy.append(digitClassifier.getAcc(pred, newlabels))

                test_labels = [recognition_model.class_to_int[c] for c in gtTuple[2]]
                junk_Testlabels = [len(recognition_model.int_to_class) for i in range(len(jTestFeat))]
                tLabels = np.array(test_labels)
                if config == '0':
                    tLabels = np.r_[test_labels, junk_Testlabels]
                testpred = digitClassifier.classify_pipeline(testfeat, recognition_model)
                print("\nTest Accuracy ->", end="")
                testAccuracy.append(digitClassifier.getAcc(testpred, tLabels))
                folderNumber += 1
                print("\n\n")

            with open('models/finetuneresults.txt', 'w+') as file:
                modelNum = 1
                for tr, de, ta, tea in zip(tuneTrees, tuneDepth, trainAccuracy, testAccuracy):
                    string = "Model --> " + str(modelNum) + " Tree:" + str(tr) + " Depth:" + str(de) + " Train Accuracy:" + str(ta) + " Test Accuracy:" + str(tea) + "\n"
                    file.write(string)
                    modelNum += 1
        else:
            print('Usage [Python] train [0/1]')