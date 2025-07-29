from mdgen.parsing import parse_train_args
args = parse_train_args()

from mdgen.dataset import MDGenDataset
from collections import defaultdict

trainset = MDGenDataset(args, split=args.train_split)
valset = MDGenDataset(args, split=args.val_split, repeat=args.val_repeat)
TEST_SPLIT = 'splits/4AA_test.csv'
MODEL_DIR = 'workdir/default'

trainset_names = sorted(trainset.df.index.tolist())
valset_names = sorted(valset.df.index.tolist())
trainset_len = len(trainset_names)
valset_len = len(valset_names)
valset_effective_len = valset_len * args.val_repeat

# Save the train and validation sets to text files
with open(MODEL_DIR + '/train_set.txt', 'w') as f:
    f.write(f'Train set length: {trainset_len}\n')
    for name in trainset_names:
        f.write(name + '\n')
with open(MODEL_DIR + '/val_set.txt', 'w') as f:  
    f.write(f'Validation set length: {valset_len}\n')
    f.write(f'Effective validation set length (after repeat): {valset_effective_len}\n')
    f.write(f'Validation set repeat factor: {args.val_repeat}\n')
    for name in valset_names:
        f.write(name + '\n')

# Val-Train bigram overlap
overlap = defaultdict(set)
for val_name in valset_names:
    val_name_bigrams = [val_name[i:i+2] for i in range(len(val_name) - 1)]
    for train_name in trainset_names:
        if any(duet in train_name for duet in val_name_bigrams):
            overlap[val_name].add(train_name)

# Save the overlap to a text file
with open(MODEL_DIR + '/val_train_bigram_overlap.txt', 'w') as f:
    f.write(f'Validation name - Train names with bigram overlap\n')
    for val_name, train_names in overlap.items():
        f.write(f'{val_name} - {sorted(list(train_names))}\n')

testset = MDGenDataset(args, split=TEST_SPLIT)
testset_names = sorted(testset.df.index.tolist())
with open(MODEL_DIR + '/test_set.txt', 'w') as f:
    f.write(f'Test set length: {len(testset_names)}\n')
    for name in testset_names:
        f.write(name + '\n')

# Test-Train bigram overlap
test_train_overlap = defaultdict(set)
for test_name in testset_names:
    test_name_bigrams = [test_name[i:i+2] for i in range(len(test_name) - 1)]
    for train_name in trainset_names:
        if any(duet in train_name for duet in test_name_bigrams):
            test_train_overlap[test_name].add(train_name)

# Save the test-train overlap to a text file
with open(MODEL_DIR + '/test_train_bigram_overlap.txt', 'w') as f:
    f.write(f'Test name - Train names with bigram overlap\n')
    for test_name, train_names in test_train_overlap.items():
        f.write(f'{test_name} - {sorted(list(train_names))}\n')