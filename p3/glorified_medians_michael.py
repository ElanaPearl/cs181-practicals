import numpy as np
import csv
import pandas as pd
from collections import defaultdict
import random

# Predict via the user-specific median.
# If the user has no data, use the global median.

train_file = 'train.csv'
test_file  = 'test.csv'
artist_file = 'artists.csv'
user_file = 'profiles.csv'


# Load the training data and split into test + validation.
def get_train_and_val_data():
    train_data = {}
    val_data = {}
    with open(train_file, 'r') as train_fh:
        train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
        next(train_csv, None)
        for row in train_csv:
            user   = row[0]
            artist = row[1]
            plays  = row[2]
        
            if random.randint(0,10) < percent_val*10:
                if not user in val_data:
                    val_data[user] = {}
                val_data[user][artist] = int(plays)
            else:
                if not user in train_data:
                    train_data[user] = {}

                train_data[user][artist] = int(plays)
    return train_data, val_data

user_data = {}
with open(user_file, 'r') as user_fh:
    user_csv = csv.reader(user_fh, delimiter=',', quotechar='"')
    next(user_csv, None)
    for row in user_csv:
        user = row[0]
        sex = row[1]
        age  = row[2]
        country = row[3]
    
        if not user in user_data:
            user_data[user] = {}
        
        user_data[user] = [sex, age, country]

# Compute the global median and per-user median.
def fit_data(train_data):
    plays_array  = []
    user_median = {}
    artist_total = {}
    artist_plays = defaultdict(list)
    artist_median = {}
    artist_total = {}
    artist_area = {}
    artist_std = {}
    user_std = {}

    for user, user_data in train_data.iteritems():
        user_plays = []
        for artist, plays in user_data.iteritems():
            plays_array.append(plays)
            user_plays.append(plays)
            artist_plays[artist].append(plays)


        user_median[user] = np.median(np.array(user_plays))
        user_std[user] = np.std(np.array(user_plays))

    for artist, plays in artist_plays.iteritems():
        artist_median[artist] = np.median(plays)
        artist_std[artist] = np.std(plays)
        artist_total[artist] = sum(plays)


    global_median = np.median(np.array(plays_array))
    return artist_median, user_median, global_median, artist_std, user_std

def predict(artist, user, artist_median, user_median, global_median, artist_std, user_std, weight): 
    if artist in artist_median:
        popularity = float(artist_median[artist])/global_median
        popularity = ((artist_median[artist] - global_median)/(artist_std[artist]) )
    else:
        popularity = 1
    if user in user_median:
        if user_median[user] + (popularity*user_std[user]*weight) < 0:
            return user_median[user]
        else:
            return user_median[user]+(popularity*user_std[user]*weight)
    else:
        return global_median


def validate(val_data, artist_median, user_median, global_median, weight):
    total_abs_error = 0
    num_preds = 0
    for user, user_data in val_data.iteritems():
        for artist, true_plays in user_data.iteritems():
            pred_plays = predict(artist, user, artist_median, user_median, global_median, artist_std, user_std, weight)
            total_abs_error += abs(pred_plays - true_plays)
            num_preds += 1
    return float(total_abs_error) / num_preds


def get_best_w(train_data, val_data, artist_median, user_median, global_median):
    w_s = np.arange(0, 2, .01)
    errs = []

    print "trying weight values"
    for i, w in enumerate(w_s):    
        err = validate(val_data, artist_median, user_median, global_median, w)
        if i % 10 == 0:
            print i, err
        errs.append(err)

    min_error = min(errs)
    best_w = w_s[np.argmin(errs)]
    return min_error, best_w


def write_predictiions(artist_median, user_median, global_median, weight):
    soln_file  = 'user_median_plus_popularity.csv'

    with open(test_file, 'r') as test_fh:
        test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
        next(test_csv, None)

        with open(soln_file, 'w') as soln_fh:
            soln_csv = csv.writer(soln_fh,
                                  delimiter=',',
                                  quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
            soln_csv.writerow(['Id', 'plays'])
            for row in test_csv:
                id     = row[0]
                user   = row[1]
                artist = row[2]

                if user in user_median:
                    pred = predict(artist, user, artist_median, user_median, global_median, artist_std, user_std, weight)
                    soln_csv.writerow([id, pred])
                else:
                    print "User", id, "not in training data."
                    if artist in artist_medians:
                        soln_csv.writerow([id, artist_median[artist]])
                    else:
                        soln_csv.writerow([id, global_median])


print "Loading data"
percent_val = .2
train_data, val_data = get_train_and_val_data()
print "Data loaded"
artist_median, user_median, global_median, artist_std, user_std = fit_data(train_data)
print "Fitting data"


print "Getting w"
err, best_w = get_best_w(train_data, val_data, artist_median, user_median, global_median)
print err, best_w


#best_w = .9

percent_val = 0.1
print "Getting data"
train_data, val_data = get_train_and_val_data()

print "Fitting data"
artist_median, user_median, global_median, artist_std, user_std = fit_data(train_data)

print "Writing predictions"
write_predictiions(artist_median, user_median, global_median, best_w)


