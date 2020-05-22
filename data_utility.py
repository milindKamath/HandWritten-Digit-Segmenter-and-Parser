import math
import numpy as np
from digitClassifier import Digit
from bs4 import BeautifulSoup


class Expression:

    def __init__(self, file, uid, strokes, symbols=[]):
        self.file = file
        self.uid = uid
        self.strokes = strokes
        self.symbols = symbols


class Symbol:

    def __init__(self, group_id, strokes, classification=None):
        self.group_id = group_id
        self.strokes = strokes
        self.classification = classification
        self.weight = 0.0

    def to_digit(self):
        return Digit(self.group_id, list(map(lambda x: np.array(x.points).astype(np.float64), self.strokes)))


class Stroke:

    def __init__(self, points, id):
        self.id = id
        self.points = points


def kl_divergance(P, Q):
    total = 0
    for p, q in zip(P, Q):
        if p != 0 and q != 0:
            total += p * math.log(p/q)
    return total


def inkml_to_Expression(path, file, ground_truth):
    with open(file, 'r') as fp:
        try:
            soup = BeautifulSoup(fp, features='lxml', from_encoding=fp.encoding)
            if soup != None:
                    tag_ui = soup.find('annotation', {'type': 'UI'})
                    strokes = {}
                    traces = soup.find_all('trace')
                    for trace in traces:
                        trace_split = trace.text.strip().split(',')
                        trace_split = list(map(lambda x: x.split()[:2], trace_split))
                        new_traces = [trace_split[0]]
                        for i in trace_split:
                            if new_traces[-1] != i:
                                new_traces.append(i)
                        trace_split = np.array(new_traces, dtype=float)
                        strokes[int(trace.attrs['id'])] = (Stroke(trace_split, int(trace.attrs['id'])))
                    symbols = []
                    if(ground_truth):
                        traceGroup = soup.find('tracegroup')
                        for symbol in traceGroup.find_all('tracegroup'):
                            group_id = symbol.attrs['xml:id']
                            classification = symbol.find('annotation', {'type': 'truth'}).text
                            symbol_strokes = []
                            for stroke_num in symbol.find_all('traceview'):
                                symbol_strokes.append(strokes[int(stroke_num.attrs['tracedataref'])])
                            symbols.append(Symbol(group_id, symbol_strokes, classification))

                    if tag_ui is not None:
                        if path[-1:] != '/':
                            path += '/'
                        return Expression(file.replace(path, ""), tag_ui.text, strokes, symbols)
        except:
            return None


def get_symbol_counts(expressions):
    counts = []
    symbol_to_index = {}
    for expression in expressions:
        for symbol in expression.symbols:
            index = symbol_to_index.get(symbol.classification, len(counts))
            if index == len(counts):
                symbol_to_index[symbol.classification] = index
                counts.append(1)
            else:
                counts[index] += 1
    return counts, symbol_to_index


def count_to_probability(arr):
    np_arr = np.array(arr)
    total = np_arr.sum()
    return np_arr / total