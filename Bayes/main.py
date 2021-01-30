from math import exp, log
from os import listdir
import matplotlib.pyplot as plot

LEGIT = 0
SPAM = 1
CLASSES = 2
PARTS = 10


class Message:

    def __init__(self, cl, words):
        self.cl = cl
        self.words = words


class Bayes:

    def __init__(self, all_words, n, class_cnt, zn, prob):
        self.all_words = all_words
        self.n = n
        self.class_cnt = class_cnt
        self.zn = zn
        self.prob = prob


def bayes_accuracy(train_msgs, test_msgs, lc, alpha):
    bayes = get_bayes(train_msgs, alpha)
    predicted = 0
    for msg in test_msgs:
        predict = get_predict(msg, bayes, alpha, lc)
        if predict == msg.cl:
            predicted += 1
    return predicted / len(test_msgs)


def get_bayes(train_msgs, alpha):
    all_words = set()
    words = [{} for _ in range(CLASSES)]
    class_cnt = [0] * CLASSES
    for msg in train_msgs:
        current_msg = set(msg.words)
        for word in msg.words:
            all_words.add(word)
        for word in current_msg:
            if word not in words[msg.cl]:
                words[msg.cl][word] = 0
            words[msg.cl][word] += 1
        class_cnt[msg.cl] += 1
    zn, prob = [], [{} for _ in range(CLASSES)]
    for cl in range(CLASSES):
        zn.append(class_cnt[cl] + alpha * 2.0)
        for word, cnt in words[cl].items():
            prob[cl][word] = (cnt + alpha) / zn[cl]
    return Bayes(all_words, len(train_msgs), class_cnt, zn, prob)


def get_bayes_result_probs(msg, bayes, alpha, lc):
    current_msg = set(msg.words)
    results = []
    for cl in range(CLASSES):
        ln_result = log(bayes.class_cnt[cl] / bayes.n)
        for word in bayes.all_words:
            current_prob = alpha / bayes.zn[cl]
            if word in bayes.prob[cl]:
                current_prob = bayes.prob[cl][word]
            if word not in current_msg:
                current_prob = 1 - current_prob
            ln_result += log(current_prob)
        ln_result += log(lc[cl])
        results.append(ln_result)
    s = sum(results)
    results[0] /= s
    results[1] /= s
    return results


def get_predict(msg, bayes, alpha, lc):
    results = get_bayes_result_probs(msg, bayes, alpha, lc)
    predict = LEGIT
    if results[SPAM] > results[LEGIT]:
        predict = SPAM
    return predict


def read_part(part):
    part_dir = "messages/part" + str(part)
    msgs = []
    for file_name in listdir(part_dir):
        cl = SPAM
        if file_name.find("legit") != -1:
            cl = LEGIT
        with open(part_dir + "/" + file_name) as f:
            subject_words = next(f)[8:].strip().split()
            _ = next(f)
            text_words = next(f).strip().split()
            msgs.append(Message(cl, subject_words + text_words))
    return msgs


def get_k_fold_step(msgs_parts, test_index):
    train_msgs = []
    for i in range(test_index):
        train_msgs += msgs_parts[i]
    for i in range(test_index + 1, PARTS):
        train_msgs += msgs_parts[i]
    return train_msgs, msgs_parts[test_index]


def k_fold_accuracy(msgs_parts, lc, alpha):
    accuracy = 0.0
    for test_index in range(PARTS):
        print(alpha, test_index)
        train_msgs, test_msgs = get_k_fold_step(msgs_parts, test_index)
        accuracy += bayes_accuracy(train_msgs, test_msgs, lc, alpha)
    return accuracy / PARTS


def k_fold_stats(msgs_parts, lc, alpha):
    accuracy = 0.0
    legit_false, spam_false = 0, 0
    for test_index in range(PARTS):
        print(alpha, test_index)
        train_msgs, test_msgs = get_k_fold_step(msgs_parts, test_index)
        accuracy += bayes_accuracy(train_msgs, test_msgs, lc, alpha)
        bayes = get_bayes(train_msgs, alpha)
        for msg in test_msgs:
            predict = get_predict(msg, bayes, alpha, lc)
            if msg.cl == LEGIT and predict == SPAM:
                legit_false += 1
            elif msg.cl == SPAM and predict == LEGIT:
                spam_false += 1
    return legit_false, spam_false, accuracy / PARTS


def convert_part_to_ngramms(original_part, n):
    return [Message(msg.cl, [" ".join(msg.words[i:i + n]) for i in range(len(msg.words) - n + 1)])
            for msg in original_part]


def draw_roc(original_parts, alpha):
    all_parts = []
    for op in original_parts:
        all_parts += op
    bayes = get_bayes(all_parts, alpha)
    all_res = []
    for msg in all_parts:
        all_res.append(get_bayes_result_probs(msg, bayes, alpha, [exp(1), exp(1)]))
    all_res.sort(key=lambda res: -res[1])
    cnt_legit, cnt_spam = 0, 0
    for msg in all_parts:
        if msg.cl == LEGIT:
            cnt_legit += 1
        else:
            cnt_spam += 1
    legit_step, spam_step = 1 / cnt_legit, 1 / cnt_spam
    current_x, current_y = 0, 0
    xx, yy = [current_x], [current_y]
    for res in all_res:
        if res[LEGIT] > res[SPAM]:
            current_x += spam_step
        else:
            current_y += legit_step
        xx.append(current_x)
        yy.append(current_y)
    plot.title("ROC")
    plot.xlabel("fp")
    plot.ylabel("tp")
    plot.plot(xx, yy)
    plot.show()


def draw_l_dependency(original_parts):
    l_spam, l_legit = 1, 1
    xx, yy = [], []
    while l_legit <= 92:
        stats = k_fold_stats(original_parts, [exp(l_legit), exp(l_spam)], 0.01)
        print(l_legit, stats)
        xx.append(l_legit)
        yy.append(stats[2])
        print(stats)
        if stats[0] == 0:
            break
        l_legit += 15

    plot.title("l_spam=1 => l_legit=92")
    plot.xlabel("l_legit")
    plot.ylabel("accuracy")
    plot.plot(xx, yy)
    plot.show()


WITH_FINDING_HYPER = True
WITH_PLOTS = False


def main():
    original_parts = [read_part(i + 1) for i in range(PARTS)]

    if WITH_FINDING_HYPER:
        lc = [exp(1), exp(1)]
        best_accuracy, best_alpha, best_n = 0.0, 0.01, 1
        for n in range(1, 4):
            parts = [convert_part_to_ngramms(op, n) for op in original_parts]
            alpha = 0.001
            while alpha <= 10:
                accuracy = k_fold_accuracy(parts, lc, alpha)
                print(n, accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_alpha = alpha
                    best_n = n
                alpha *= 10
        print("Best accuracy =", best_accuracy,
              "\tn =", best_n,
              "\talpha =", best_alpha)

    if WITH_PLOTS:
        draw_l_dependency(original_parts)
        parts2gramm = [convert_part_to_ngramms(op, 2) for op in original_parts]
        best_alpha = 0.001
        draw_roc(parts2gramm, best_alpha)


if __name__ == '__main__':
    main()


