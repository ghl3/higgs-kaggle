
from collections import defaultdict


def score_summary(classifier, features, targets, scoring, **kwargs):
    print "Using %s features and %s rows\n" % (len(features.columns), len(features))
    for cv in scoring:
        scores = cross_validation.cross_val_score(classifier, features, targets, scoring=cv, **kwargs)
        print '----- %s -----' % cv
        print scores
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print '\n'


def importance_summary(fitted, features):
    return sorted([(feature, importance) for (feature, importance) in zip(features.columns, fitted.feature_importances_)], 
           key = lambda x: -x[1])


def create_solution_dictionary(soln):
    """ Read solution file, return a dictionary with key EventId and value (weight,label).
    Solution file headers: EventId, Label, Weight """

    solnDict = {}
    for index, row in soln.iterrows():
        if row[0] not in solnDict:
            solnDict[index] = (row['Label'], row['Weight'])
    return solnDict


def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )
    where b_r = 10, b = background, s = signal, log is natural logarithm """

    br = 10.0
    radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print 'radicand is negative. Exiting'
        exit()
    else:
        return math.sqrt(radicand)


def ams_metric_threshold(submission, truth_dict, threshold=0.5, verbose=False, confusion=False):
    """  Prints the AMS metric value to screen.
    Solution File header: EventId, Class, Weight
    Submission File header: EventId, RankOrder, Class
    """

    if confusion:
        confusion_matrix = defaultdict(lambda: 0)

    signal = 0.0
    background = 0.0
    for index, row in submission.iterrows():

        score = row['predict_proba_1']
        classification = target_to_label(score>threshold)
        try:
            truth = truth_dict[index][0]
        except:
            print "index %s not in signal dict" % index
            continue
        if verbose:
            print "Index: %s Score: %s Classification: %s Truth: %s" % (index, score, classification, truth)

        if confusion:
            confusion_matrix[(classification, truth)] += 1

        if classification == 's': # only events predicted to be signal are scored
            if truth == 's':
                signal += float(truth_dict[index][1])
            elif truth == 'b':
                background += float(truth_dict[index][1])
            else:
                print "WTF"

    ams = AMS(signal, background)

    if verbose:
        print 'signal = {0}, background = {1}'.format(signal, background)
        print 'AMS = ' + str(ams)
        if confusion:
            for key, val in confusion_matrix.iteritems():
                print key, val

    return ams


def plot_ams(prediction, truth_dict):
    thresholds = np.arange(0, 1, .05)
    metrics = [AMS_metric_threshold(prediction, truth_dict, threshold) for threshold in thresholds]

    max_ams = max(metrics)
    plt.plot(thresholds, metrics, label='AMS vs threshold (max = %0.2f)' % max_ams)
    plt.xlabel('Threshold')
    plt.ylabel('AMS Metric')
    plt.legend(loc="upper left")


