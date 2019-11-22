import numpy as np
import matplotlib.pyplot as plt
from extract_features import extract_features, convert_features
from plots import plot_feature_stats, plot_confusion_matrix, plot_learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report


def train_decision_tree(train_feats, train_labels):

    dec_tree = DecisionTreeClassifier(random_state=420)
    parameters = {'min_samples_split': [100, 200, 500], 
                'min_samples_leaf': [200, 500, 1000]}
    classifier = GridSearchCV(dec_tree, parameters, 
        scoring=make_scorer(accuracy_score), verbose=1, iid=False, n_jobs=4, cv=4)
    classifier.fit(train_feats, train_labels)
    cv_results = classifier.cv_results_
    print('Fit time: {0:.2f}'.format(np.sum(cv_results['mean_fit_time'])))
    print('Refit time: {0:.2f}'.format(classifier.refit_time_))
    print(classifier.best_params_)

    return classifier.best_estimator_

def train_neural_network(train_feats, train_labels):

    fnn = MLPClassifier(hidden_layer_sizes=(120, 80), 
        max_iter=600, tol=1e-5, early_stopping=False, n_iter_no_change=10, 
        learning_rate='adaptive', learning_rate_init=1e-3, alpha=1e-4,
        verbose=True, random_state=420)
    
    # parameters = {'alpha': [1e-3, 1e-4, 1e-5],
    #             'learning_rate_init': [1e-2, 1e-3, 1e-4]}
    # classifier = GridSearchCV(fnn, parameters, 
    #     scoring=make_scorer(accuracy_score), verbose=1, iid=False, n_jobs=4, cv=2)
    # classifier.fit(train_feats, train_labels)
    # cv_results = classifier.cv_results_
    # print('Fit time: {0:.2f}'.format(np.sum(cv_results['mean_fit_time'])))
    # print('Refit time: {0:.2f}'.format(classifier.refit_time_))
    # print(classifier.best_params_)
    # return classifier.best_estimator_

    classifier = fnn
    classifier.fit(train_feats, train_labels)
    
    return classifier

def save_parameters(model_path, weights, biases):

    np.save(model_path + '_weights.npy', weights, allow_pickle=True)
    np.save(model_path + '_biases.npy', biases, allow_pickle=True)

    for i, (w, b) in enumerate(zip(weights, biases)):
        
        np.savetxt(model_path + '_weights' + str(i) + '.gz', w)
        np.savetxt(model_path + '_biases' + str(i) + '.gz', b)


def main():

    data_path = '../data/'
    model_path = '../model_6/model_6'
    mod_types = ['gfsk', 'gmsk', 'qam4', 'qam16', 'qam64', 'psk2', 'psk4', 'psk8']
    feature_types = ['cumulants', 'amplitude_stats', 'phase_stats']
    # feature_types = ['amplitude_stats', 'phase_stats']
    feature_names = ['|C20|', '|C21|', '|C40|', '|C41|', '|C42|', '|C60|', '|C61|', 
        '|C62|', '|C63|', '∠C20', '∠C21', '∠C40', '∠C41', '∠C42', '∠C60', '∠C61', 
        '∠C62', '∠C63', 'Magnitude mean', 'Magnitude std', 'Phase mean', 'Phase std']

    num_frames = int(1e3)
    frame_len = 2048
    frame_step = 256
    snr_list = None

    files = dict()
    for mod_type in mod_types:

        files[mod_type] = list()
        files[mod_type].append(data_path + mod_type + '.txt')
 
    print('Extracting features')
    features_dict = extract_features(files, mod_types, feature_types=feature_types, 
        frame_len=frame_len, frame_step=frame_step, num_frames=num_frames, 
        snr_list=snr_list, verbose=True)
    plot_feature_stats(features_dict, feature_names)
    print('Number of features: {0}'.format(features_dict[mod_types[0]].shape[1]))

    print('Converting features')
    (features, labels) = convert_features(features_dict)
    (train_feats, test_feats, train_labels, test_labels) = train_test_split(features, labels, test_size=0.1)
    print('Number of samples: {0}'.format(features.shape[0]))
    print('Number of training samples: {0}'.format(train_feats.shape[0]))
    print('Number of testing samples: {0}'.format(test_feats.shape[0]))

    print('Training model')
    classifier = train_neural_network(train_feats, train_labels)

    print('Saving parameters')
    save_parameters(model_path, classifier.coefs_, classifier.intercepts_)

    print('Testing model')
    pred_labels = classifier.predict(test_feats)
    plot_confusion_matrix(test_labels, pred_labels, np.array(mod_types), 
        title='', normalize=True)
    # plot_learning_curve(classifier, train_feats, train_labels, 
    #     title='Learning Curve', cv=3, n_jobs=2, train_sizes=np.linspace(0.1, 1.0, 10))

    accuracy = accuracy_score(test_labels, pred_labels)
    print('Accuracy: {:.2f}'.format(accuracy))

    report = classification_report(test_labels, pred_labels, target_names=mod_types)
    print(report)

    print('Saving model')
    import pickle
    with open(model_path + '.pkl', 'wb') as file:
        pickle.dump(classifier, file)

    plt.show()

if __name__ == '__main__':

    main()
