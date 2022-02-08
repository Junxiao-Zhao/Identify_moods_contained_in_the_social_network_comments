import thulac, csv, numpy as np

def classify(test, p0, p1, p2, p3):
    l = [sum(test*p0), sum(test*p1), sum(test*p2), sum(test*p3)]
    closest = max(l)
    return l.index(closest)

def train(matrix, category):
    print("Training...")
    num_words = len(matrix[0])
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p2_num = np.ones(num_words)
    p3_num = np.ones(num_words)
    p0_denom = 2
    p1_denom = 2
    p2_denom = 2
    p3_denom = 2
    for i in range(len(matrix)):
        if (category[i] == 0):
            p0_num += matrix[i]
            p0_denom += sum(matrix[i])
        if (category[i] == 1):
            p1_num += matrix[i]
            p1_denom += sum(matrix[i])
        if (category[i] == 2):
            p2_num += matrix[i]
            p2_denom += sum(matrix[i])
        else:
            p3_num += matrix[i]
            p3_denom += sum(matrix[i])
    p0 = p0_num / p0_denom
    p1 = p1_num / p1_denom
    p2 = p2_num / p2_denom
    p3 = p3_num / p3_denom
    return p0, p1, p2, p3


def match(vocab, text):
    l = [0] * len(vocab)
    for words in text:
        if words in vocab:
            l[vocab.index(words)] = 1
        else:
            print("The word: %s is not in my Vocabulary!" % words)
    return l

def nodup(data):
    print("Eliminating duplicate words...")
    vocab = set([])
    for document in data:
        vocab = vocab | set(document)
    return list(vocab)

def read_csv(filename):
    print("Reading files...")
    thulac.thu1 = thulac.thulac(seg_only=True)
    data = list()
    category = list()
    with open(filename, 'r', encoding='utf-8-sig') as csvFile:
        reader = list(csv.reader(csvFile))
        for lines in reader[1:]:
            category.append(int(lines[0]))
            data.append(thulac.thu1.cut(lines[1][1:], text=True).split(" "))
    return data, category

def main():
    data, category = read_csv("simplifyweibo_4_moods.csv")
    num_text = len(data)
    split_index = int(num_text*0.9)
    vocab = nodup(data[:split_index])
    matrix = []
    for texts in data[:split_index]:
        matrix.append(match(vocab, texts))
    p0, p1, p2, p3 = train(np.array(matrix), np.array(category[:split_index]))
    print(p0)
    print(p1)
    print(p2)
    print(p3)
    for i in range(split_index, num_text-1):
        l = np.array(match(vocab, data[i]))
        print("Text #%d's label is %d and is classified as %d" % (i+1, category[i], classify(l, p0, p1, p2, p3)))

main()