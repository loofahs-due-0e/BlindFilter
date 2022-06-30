import math
import numpy as np
import sys

dataset = 'enron'

if len(sys.argv) != 2:
    print("usage : python3 train.py [ train_size in % (recommend 1 or 10) ]")
    exit()

train_size = float(sys.argv[1]) * 0.005
val_size = train_size
test_size = 1. - train_size - val_size

# assert train_size + val_size + test_size == 1.0

with open(dataset + "/ham_index", 'r') as f:
    allHam_index = f.readlines()
with open(dataset + "/spam_index", 'r') as f:
    allSpam_index = f.readlines()

allHam_index = [set(map(int, e.rstrip('\n').split(' '))) for e in allHam_index]
allSpam_index = [set(map(int, e.rstrip('\n').split(' ')))
                 for e in allSpam_index]

num_allHam = len(allHam_index)
num_allSpam = len(allSpam_index)
print(f'Dataset:  {dataset}')
print(f'HAM:      {num_allHam}')
print(f'SPAM:     {num_allSpam}')

num_testHam = int(np.round(num_allHam * test_size))
num_validHam = int(np.round(num_allHam * val_size))
num_trainHam = num_allHam - num_testHam - num_validHam
num_testSpam = int(np.round(num_allSpam * test_size))
num_validSpam = int(np.round(num_allSpam * val_size))
num_trainSpam = num_allSpam - num_testSpam - num_validSpam

# Shuffle and Divide Dataset
np.random.shuffle(allHam_index)
np.random.shuffle(allSpam_index)

trainHam_index = allHam_index[:num_trainHam]
testHam_index = allHam_index[num_trainHam: num_trainHam + num_testHam]
validHam_index = allHam_index[num_trainHam + num_testHam:]
trainSpam_index = allSpam_index[:num_trainSpam]
testSpam_index = allSpam_index[num_trainSpam: num_trainSpam + num_testSpam]
validSpam_index = allSpam_index[num_trainSpam + num_testSpam:]

print('          Train    Test   Validation')
print(f'Ham:      {num_trainHam}     {num_testHam}     {num_validHam}')
print(f'Spam:     {num_trainSpam}     {num_testSpam}     {num_validSpam}')


'''Get Likelihood Probability'''
token_likelihood_ham = []  # p(t|ham)
token_likelihood_spam = []  # p(t|spam)

num_trainingHam = len(trainHam_index)
num_trainingSpam = len(trainSpam_index)
num_train = num_trainingHam + num_trainingSpam

p_ham = (num_trainingHam + 1) / (num_train + 2)
p_spam = (num_trainingSpam + 1) / (num_train + 2)

for i in range(30522):      # 30522 tokens
    num_token_ham = sum([1 if i in e else 0 for e in trainHam_index])
    num_token_spam = sum([1 if i in e else 0 for e in trainSpam_index])
    token_likelihood_ham.append((num_token_ham+1) / (num_trainingHam+2))
    token_likelihood_spam.append((num_token_spam+1) / (num_trainingSpam+2))

    if i % 5000 == 0 and i != 0:
        print(f'{i}th Done')

''' save prob p by logp'''
with open("probdiff", "w") as f:
    for i in range(30522):
        value = math.log(
            token_likelihood_ham[i]) - math.log(token_likelihood_spam[i])
        if i != 0:
            f.write('\n')
        f.write(str(value))

'''Find Best Threshold with Validation Set'''
num_validationHam = len(validHam_index)
num_validationSpam = len(validSpam_index)
num_valid = num_validationHam + num_validationSpam

best_largest_threshold = -1
best_smallest_threshold = -1
best_accuracy = -1
best_accuracy_ham = -1
best_accuracy_spam = -1

threshold_start = 0.9
for threshold in np.arange(threshold_start, 1, 0.001):
    num_correct = 0
    num_correct_ham = 0
    num_correct_spam = 0

    for ham in validHam_index:
        result = sum(
            [math.log(token_likelihood_ham[token]) -
             math.log(token_likelihood_spam[token])
             for token in ham]) + math.log(p_ham) - math.log(p_spam)
        if result >= math.log(1/threshold-1):
            num_correct += 1
            num_correct_ham += 1

    for spam in validSpam_index:
        result = sum(
            [math.log(token_likelihood_ham[token]) -
             math.log(token_likelihood_spam[token])
             for token in spam]) + math.log(p_ham) - math.log(p_spam)
        if result < math.log(1/threshold-1):
            num_correct += 1
            num_correct_spam += 1

    accuracy = num_correct / num_valid
    if accuracy >= best_accuracy:
        best_largest_threshold = threshold
        if accuracy > best_accuracy:
            best_smallest_threshold = threshold

        best_accuracy = accuracy

        accuracy_ham = num_correct_ham / num_validationHam
        accuracy_spam = num_correct_spam / num_validationSpam
        best_accuracy_ham = accuracy_ham
        best_accuracy_spam = accuracy_spam


print(f'Best accuracy: {best_accuracy}')
print(f'Threshold: {best_smallest_threshold} - {best_largest_threshold}')
print(f'Best accuracy HAM: {best_accuracy_ham}')
print(f'Best accuracy SPAM: {best_accuracy_spam}')


'''Verify Test Email'''
best_threshold = best_largest_threshold

num_testingHam = len(testHam_index)
num_testingSpam = len(testSpam_index)
num_test = num_testingHam + num_testingSpam

num_correct = 0

num_correct_ham = 0
num_correct_spam = 0

for ham in testHam_index:
    result = sum(
        [math.log(token_likelihood_ham[token]) -
         math.log(token_likelihood_spam[token])
         for token in ham]) + math.log(p_ham) - math.log(p_spam)
    # result = log(1/p-1)
    if result >= math.log(1/best_threshold - 1):
        num_correct += 1
        num_correct_ham += 1

for spam in testSpam_index:
    result = sum(
        [math.log(token_likelihood_ham[token]) -
         math.log(token_likelihood_spam[token])
         for token in spam]) + math.log(p_ham) - math.log(p_spam)
    # result = log(1/p-1)
    if result < math.log(1/best_threshold - 1):
        num_correct += 1
        num_correct_spam += 1

accuracy = num_correct / num_test
TP = num_correct_ham
FP = num_testingHam - num_correct_ham
FN = num_testingSpam - num_correct_spam
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2*(recall * precision) / (recall + precision)
print(f'Threshold: {best_threshold}')
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1_score}')
