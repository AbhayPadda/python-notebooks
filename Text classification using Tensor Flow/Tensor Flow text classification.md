

```python
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import string
import unicodedata
import sys
from tflearn.data_utils import to_categorical, pad_sequences

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

# method to remove punctuations from sentences.
def remove_punctuation(text):
    no_punct = ""
    for char in text:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct

# initialize the stemmer
stemmer = LancasterStemmer()
```


```python
from io import StringIO
import requests
import json
import pandas as pd

df_data_1 = pd.read_csv(get_object_storage_file_with_credentials_2659b32ff9a04774afc2ee0815088bcf('project1', 'Stream_Twitter.csv'))
print(df_data_1.head())

print(df_data_1.shape)

print(df_data_1.groupby(by=['SPAM'])['SPAM'].count())

```

       S. No.  SPAM                                    Sound Bite Text
    0       1     1  RT @foxyCAMS: Watch live HOT shows join for fr...
    1       2     1  Watch live HOT shows, join for free ? camslut....
    2       3     1  Watch live HOT shows, join for free ? camslut....
    3       4     1  Sexy camgirls are hot to hard free camshow ? b...
    4       5     1  RT @twitxgnd: pretty pussy #pussy #prettypussy...
    (10814, 3)
    SPAM
    0    5962
    1    4852
    Name: SPAM, dtype: int64



```python
labels    = df_data_1['SPAM']
sentences = df_data_1['Sound Bite Text']

words = []
# a list of tuples with words in the sentence and category name
docs = []

categories = list([0,1])
```


```python
rowNum = 0;

for text in sentences:
    each_sentence = remove_punctuation(text)
    w = nltk.word_tokenize(each_sentence)
    words.extend(w)
    docs.append((w, labels[rowNum]))
    rowNum = rowNum + 1
    
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))
```


```python
# create our training data
training = []
output = []
# create an empty array for our output, taking 2 as we only have two categories
output_empty = [0] * 2


for doc in docs:
    # initialize our bag of words(bow) for each document in the list
    bow = []
    # list of tokenized words for the pattern
    token_words = doc[0]
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # create our bag of words array
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)
        
    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    # our training set will contain a the bag of words model and the output row that tells
    # which catefory that bow belongs to.
    training.append([bow, output_row])
```


```python
# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)

# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:, 0])
train_y = list(training[:, 1])
```


```python
# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=100, batch_size=8, show_metric=True)
model.save('model.tflearn')
```

    Training Step: 135199  | total loss: [1m[32m0.00211[0m[0m | time: 4.112s
    | Adam | epoch: 100 | loss: 0.00211 - acc: 0.9988 -- iter: 10808/10814
    Training Step: 135200  | total loss: [1m[32m0.00190[0m[0m | time: 4.115s
    | Adam | epoch: 100 | loss: 0.00190 - acc: 0.9989 -- iter: 10814/10814
    --
    INFO:tensorflow:/gpfs/global_fs01/cluster/yp-spark-lon02-env5-0101.bluemix.net/user/sfe5-7fdf5ebe7e8062-7f5d27323682/notebook/work/model.tflearn is not in all_model_checkpoint_paths. Manually adding it.



```python
def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))
```


```python

df_data_3 = pd.read_csv(get_object_storage_file_with_credentials_2659b32ff9a04774afc2ee0815088bcf('project1', 'TestDataSet.csv'))
df_data_3.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>_id</th>
      <th>digest</th>
      <th>spam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>V2Ut8i4ttFeDJHtx</td>
      <td>🔹🔸🔹🔸🔹🔸🔹.\n#Snapindia #indianstreetphotography ...</td>
      <td>news</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1bDQZK2ecPXzNGg5O</td>
      <td>The only reason you both think we need dairy i...</td>
      <td>social</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1lcZR50umJrhv88@s</td>
      <td>Swing by @ketsourinemacarons to cool down with...</td>
      <td>news</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1HjyxJyVN7DMT1yuy</td>
      <td>Made Ochazuke (rice and fish dish with tea pou...</td>
      <td>TPA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>WXSuvsvFr2j5SKF@</td>
      <td>You'll lose weight, have more energy, and help...</td>
      <td>news</td>
    </tr>
  </tbody>
</table>
</div>




```python
prediction = [0] * 2
test_results = list()
i = 1

for row in df_data_3['digest']:
    prediction = model.predict([get_tf_record(row)])
    test_results.append([prediction[0][0], prediction[0][1], (row.encode('utf-8'))])
    print('processing row no: ', i)
    i = i + 1
    
df = pd.DataFrame(test_results, columns = ['No Spam Prob.', 'Spam Prob.', 'Text'])
```

    processing row no:  1
    processing row no:  2
    processing row no:  3
    processing row no:  4
    processing row no:  5
    processing row no:  6
    processing row no:  7
    processing row no:  8
    processing row no:  9
    processing row no:  10
    processing row no:  11
    processing row no:  12
    processing row no:  13
    processing row no:  14
    processing row no:  15
    processing row no:  16
    processing row no:  17
    processing row no:  18
    processing row no:  19
    processing row no:  20
    processing row no:  21
    processing row no:  22
    processing row no:  23
    processing row no:  24
    processing row no:  25
    processing row no:  26
    processing row no:  27
    processing row no:  28
    processing row no:  29
    processing row no:  30
    processing row no:  31
    processing row no:  32
    processing row no:  33
    processing row no:  34
    processing row no:  35
    processing row no:  36
    processing row no:  37
    processing row no:  38
    processing row no:  39
    processing row no:  40
    processing row no:  41
    processing row no:  42
    processing row no:  43
    processing row no:  44
    processing row no:  45
    processing row no:  46
    processing row no:  47
    processing row no:  48
    processing row no:  49
    processing row no:  50
    processing row no:  51
    processing row no:  52
    processing row no:  53
    processing row no:  54
    processing row no:  55
    processing row no:  56
    processing row no:  57
    processing row no:  58
    processing row no:  59
    processing row no:  60
    processing row no:  61
    processing row no:  62
    processing row no:  63
    processing row no:  64
    processing row no:  65
    processing row no:  66
    processing row no:  67
    processing row no:  68
    processing row no:  69
    processing row no:  70
    processing row no:  71
    processing row no:  72
    processing row no:  73
    processing row no:  74
    processing row no:  75
    processing row no:  76
    processing row no:  77
    processing row no:  78
    processing row no:  79
    processing row no:  80
    processing row no:  81
    processing row no:  82
    processing row no:  83
    processing row no:  84
    processing row no:  85
    processing row no:  86
    processing row no:  87
    processing row no:  88
    processing row no:  89
    processing row no:  90
    processing row no:  91
    processing row no:  92
    processing row no:  93
    processing row no:  94
    processing row no:  95
    processing row no:  96
    processing row no:  97
    processing row no:  98
    processing row no:  99
    processing row no:  100
    processing row no:  101
    processing row no:  102
    processing row no:  103
    processing row no:  104
    processing row no:  105
    processing row no:  106
    processing row no:  107
    processing row no:  108
    processing row no:  109
    processing row no:  110
    processing row no:  111
    processing row no:  112
    processing row no:  113
    processing row no:  114
    processing row no:  115
    processing row no:  116
    processing row no:  117
    processing row no:  118
    processing row no:  119
    processing row no:  120
    processing row no:  121
    processing row no:  122
    processing row no:  123
    processing row no:  124
    processing row no:  125
    processing row no:  126
    processing row no:  127
    processing row no:  128
    processing row no:  129
    processing row no:  130
    processing row no:  131
    processing row no:  132
    processing row no:  133
    processing row no:  134
    processing row no:  135
    processing row no:  136
    processing row no:  137
    processing row no:  138
    processing row no:  139
    processing row no:  140
    processing row no:  141
    processing row no:  142
    processing row no:  143
    processing row no:  144
    processing row no:  145
    processing row no:  146
    processing row no:  147
    processing row no:  148
    processing row no:  149
    processing row no:  150
    processing row no:  151
    processing row no:  152
    processing row no:  153
    processing row no:  154
    processing row no:  155
    processing row no:  156
    processing row no:  157
    processing row no:  158
    processing row no:  159
    processing row no:  160
    processing row no:  161
    processing row no:  162
    processing row no:  163
    processing row no:  164
    processing row no:  165
    processing row no:  166
    processing row no:  167
    processing row no:  168
    processing row no:  169
    processing row no:  170
    processing row no:  171
    processing row no:  172
    processing row no:  173
    processing row no:  174
    processing row no:  175
    processing row no:  176
    processing row no:  177
    processing row no:  178
    processing row no:  179
    processing row no:  180
    processing row no:  181
    processing row no:  182
    processing row no:  183
    processing row no:  184
    processing row no:  185
    processing row no:  186
    processing row no:  187
    processing row no:  188
    processing row no:  189
    processing row no:  190
    processing row no:  191
    processing row no:  192
    processing row no:  193
    processing row no:  194
    processing row no:  195
    processing row no:  196
    processing row no:  197
    processing row no:  198
    processing row no:  199
    processing row no:  200
    processing row no:  201
    processing row no:  202
    processing row no:  203
    processing row no:  204
    processing row no:  205
    processing row no:  206
    processing row no:  207
    processing row no:  208
    processing row no:  209
    processing row no:  210
    processing row no:  211
    processing row no:  212
    processing row no:  213
    processing row no:  214
    processing row no:  215
    processing row no:  216
    processing row no:  217
    processing row no:  218
    processing row no:  219
    processing row no:  220
    processing row no:  221
    processing row no:  222
    processing row no:  223
    processing row no:  224
    processing row no:  225
    processing row no:  226
    processing row no:  227
    processing row no:  228
    processing row no:  229
    processing row no:  230
    processing row no:  231
    processing row no:  232
    processing row no:  233
    processing row no:  234
    processing row no:  235
    processing row no:  236
    processing row no:  237
    processing row no:  238
    processing row no:  239
    processing row no:  240
    processing row no:  241
    processing row no:  242
    processing row no:  243
    processing row no:  244
    processing row no:  245
    processing row no:  246
    processing row no:  247
    processing row no:  248
    processing row no:  249
    processing row no:  250
    processing row no:  251
    processing row no:  252
    processing row no:  253
    processing row no:  254
    processing row no:  255
    processing row no:  256
    processing row no:  257
    processing row no:  258
    processing row no:  259
    processing row no:  260
    processing row no:  261
    processing row no:  262
    processing row no:  263
    processing row no:  264
    processing row no:  265
    processing row no:  266
    processing row no:  267
    processing row no:  268
    processing row no:  269
    processing row no:  270
    processing row no:  271
    processing row no:  272
    processing row no:  273
    processing row no:  274
    processing row no:  275
    processing row no:  276
    processing row no:  277
    processing row no:  278
    processing row no:  279
    processing row no:  280
    processing row no:  281
    processing row no:  282
    processing row no:  283
    processing row no:  284
    processing row no:  285
    processing row no:  286
    processing row no:  287
    processing row no:  288
    processing row no:  289
    processing row no:  290
    processing row no:  291
    processing row no:  292
    processing row no:  293
    processing row no:  294
    processing row no:  295
    processing row no:  296
    processing row no:  297
    processing row no:  298
    processing row no:  299
    processing row no:  300
    processing row no:  301
    processing row no:  302
    processing row no:  303
    processing row no:  304
    processing row no:  305
    processing row no:  306
    processing row no:  307
    processing row no:  308
    processing row no:  309
    processing row no:  310
    processing row no:  311
    processing row no:  312
    processing row no:  313
    processing row no:  314
    processing row no:  315
    processing row no:  316
    processing row no:  317
    processing row no:  318
    processing row no:  319
    processing row no:  320
    processing row no:  321
    processing row no:  322
    processing row no:  323
    processing row no:  324
    processing row no:  325
    processing row no:  326
    processing row no:  327
    processing row no:  328
    processing row no:  329
    processing row no:  330
    processing row no:  331
    processing row no:  332
    processing row no:  333
    processing row no:  334
    processing row no:  335
    processing row no:  336
    processing row no:  337
    processing row no:  338
    processing row no:  339
    processing row no:  340
    processing row no:  341
    processing row no:  342
    processing row no:  343
    processing row no:  344
    processing row no:  345
    processing row no:  346
    processing row no:  347
    processing row no:  348
    processing row no:  349
    processing row no:  350
    processing row no:  351
    processing row no:  352
    processing row no:  353
    processing row no:  354
    processing row no:  355
    processing row no:  356
    processing row no:  357
    processing row no:  358
    processing row no:  359
    processing row no:  360
    processing row no:  361
    processing row no:  362
    processing row no:  363
    processing row no:  364
    processing row no:  365
    processing row no:  366
    processing row no:  367
    processing row no:  368
    processing row no:  369
    processing row no:  370
    processing row no:  371
    processing row no:  372
    processing row no:  373
    processing row no:  374
    processing row no:  375
    processing row no:  376
    processing row no:  377
    processing row no:  378
    processing row no:  379
    processing row no:  380
    processing row no:  381
    processing row no:  382
    processing row no:  383
    processing row no:  384
    processing row no:  385
    processing row no:  386
    processing row no:  387
    processing row no:  388
    processing row no:  389
    processing row no:  390
    processing row no:  391
    processing row no:  392
    processing row no:  393
    processing row no:  394
    processing row no:  395
    processing row no:  396
    processing row no:  397
    processing row no:  398
    processing row no:  399
    processing row no:  400
    processing row no:  401
    processing row no:  402
    processing row no:  403
    processing row no:  404
    processing row no:  405
    processing row no:  406
    processing row no:  407
    processing row no:  408
    processing row no:  409
    processing row no:  410
    processing row no:  411
    processing row no:  412
    processing row no:  413
    processing row no:  414
    processing row no:  415
    processing row no:  416
    processing row no:  417
    processing row no:  418
    processing row no:  419
    processing row no:  420
    processing row no:  421
    processing row no:  422
    processing row no:  423
    processing row no:  424
    processing row no:  425
    processing row no:  426
    processing row no:  427
    processing row no:  428
    processing row no:  429
    processing row no:  430
    processing row no:  431
    processing row no:  432
    processing row no:  433
    processing row no:  434
    processing row no:  435
    processing row no:  436
    processing row no:  437
    processing row no:  438
    processing row no:  439
    processing row no:  440
    processing row no:  441
    processing row no:  442
    processing row no:  443
    processing row no:  444
    processing row no:  445
    processing row no:  446
    processing row no:  447
    processing row no:  448
    processing row no:  449
    processing row no:  450
    processing row no:  451
    processing row no:  452
    processing row no:  453
    processing row no:  454
    processing row no:  455
    processing row no:  456
    processing row no:  457
    processing row no:  458
    processing row no:  459
    processing row no:  460
    processing row no:  461
    processing row no:  462
    processing row no:  463
    processing row no:  464
    processing row no:  465
    processing row no:  466
    processing row no:  467
    processing row no:  468
    processing row no:  469
    processing row no:  470
    processing row no:  471
    processing row no:  472
    processing row no:  473
    processing row no:  474
    processing row no:  475
    processing row no:  476
    processing row no:  477
    processing row no:  478
    processing row no:  479
    processing row no:  480
    processing row no:  481
    processing row no:  482
    processing row no:  483
    processing row no:  484
    processing row no:  485
    processing row no:  486
    processing row no:  487
    processing row no:  488
    processing row no:  489
    processing row no:  490
    processing row no:  491
    processing row no:  492
    processing row no:  493
    processing row no:  494
    processing row no:  495
    processing row no:  496
    processing row no:  497
    processing row no:  498
    processing row no:  499
    processing row no:  500
    processing row no:  501
    processing row no:  502
    processing row no:  503
    processing row no:  504
    processing row no:  505
    processing row no:  506
    processing row no:  507
    processing row no:  508
    processing row no:  509
    processing row no:  510
    processing row no:  511
    processing row no:  512
    processing row no:  513
    processing row no:  514
    processing row no:  515
    processing row no:  516
    processing row no:  517
    processing row no:  518
    processing row no:  519
    processing row no:  520
    processing row no:  521
    processing row no:  522
    processing row no:  523
    processing row no:  524
    processing row no:  525
    processing row no:  526
    processing row no:  527
    processing row no:  528
    processing row no:  529
    processing row no:  530
    processing row no:  531
    processing row no:  532
    processing row no:  533
    processing row no:  534
    processing row no:  535
    processing row no:  536
    processing row no:  537
    processing row no:  538
    processing row no:  539
    processing row no:  540
    processing row no:  541
    processing row no:  542
    processing row no:  543
    processing row no:  544
    processing row no:  545
    processing row no:  546
    processing row no:  547
    processing row no:  548
    processing row no:  549
    processing row no:  550
    processing row no:  551
    processing row no:  552
    processing row no:  553
    processing row no:  554
    processing row no:  555
    processing row no:  556
    processing row no:  557
    processing row no:  558
    processing row no:  559
    processing row no:  560
    processing row no:  561
    processing row no:  562
    processing row no:  563
    processing row no:  564
    processing row no:  565
    processing row no:  566
    processing row no:  567
    processing row no:  568
    processing row no:  569
    processing row no:  570
    processing row no:  571
    processing row no:  572
    processing row no:  573
    processing row no:  574
    processing row no:  575
    processing row no:  576
    processing row no:  577
    processing row no:  578
    processing row no:  579
    processing row no:  580
    processing row no:  581
    processing row no:  582
    processing row no:  583
    processing row no:  584
    processing row no:  585
    processing row no:  586
    processing row no:  587
    processing row no:  588
    processing row no:  589
    processing row no:  590
    processing row no:  591
    processing row no:  592
    processing row no:  593
    processing row no:  594
    processing row no:  595
    processing row no:  596
    processing row no:  597
    processing row no:  598
    processing row no:  599
    processing row no:  600
    processing row no:  601
    processing row no:  602
    processing row no:  603
    processing row no:  604
    processing row no:  605
    processing row no:  606
    processing row no:  607
    processing row no:  608
    processing row no:  609
    processing row no:  610
    processing row no:  611
    processing row no:  612
    processing row no:  613
    processing row no:  614
    processing row no:  615
    processing row no:  616
    processing row no:  617
    processing row no:  618
    processing row no:  619
    processing row no:  620
    processing row no:  621
    processing row no:  622
    processing row no:  623
    processing row no:  624
    processing row no:  625
    processing row no:  626
    processing row no:  627
    processing row no:  628
    processing row no:  629
    processing row no:  630
    processing row no:  631
    processing row no:  632
    processing row no:  633
    processing row no:  634
    processing row no:  635
    processing row no:  636
    processing row no:  637
    processing row no:  638
    processing row no:  639
    processing row no:  640
    processing row no:  641
    processing row no:  642
    processing row no:  643
    processing row no:  644
    processing row no:  645
    processing row no:  646
    processing row no:  647
    processing row no:  648
    processing row no:  649
    processing row no:  650
    processing row no:  651
    processing row no:  652
    processing row no:  653
    processing row no:  654
    processing row no:  655
    processing row no:  656
    processing row no:  657
    processing row no:  658
    processing row no:  659
    processing row no:  660
    processing row no:  661
    processing row no:  662
    processing row no:  663
    processing row no:  664
    processing row no:  665
    processing row no:  666
    processing row no:  667
    processing row no:  668
    processing row no:  669
    processing row no:  670
    processing row no:  671
    processing row no:  672
    processing row no:  673
    processing row no:  674
    processing row no:  675
    processing row no:  676
    processing row no:  677
    processing row no:  678
    processing row no:  679
    processing row no:  680
    processing row no:  681
    processing row no:  682
    processing row no:  683
    processing row no:  684
    processing row no:  685
    processing row no:  686
    processing row no:  687
    processing row no:  688
    processing row no:  689
    processing row no:  690
    processing row no:  691
    processing row no:  692
    processing row no:  693
    processing row no:  694
    processing row no:  695
    processing row no:  696
    processing row no:  697
    processing row no:  698
    processing row no:  699
    processing row no:  700
    processing row no:  701
    processing row no:  702
    processing row no:  703
    processing row no:  704
    processing row no:  705
    processing row no:  706
    processing row no:  707
    processing row no:  708
    processing row no:  709
    processing row no:  710
    processing row no:  711
    processing row no:  712
    processing row no:  713
    processing row no:  714
    processing row no:  715
    processing row no:  716
    processing row no:  717
    processing row no:  718
    processing row no:  719
    processing row no:  720
    processing row no:  721
    processing row no:  722
    processing row no:  723
    processing row no:  724
    processing row no:  725
    processing row no:  726
    processing row no:  727
    processing row no:  728
    processing row no:  729
    processing row no:  730
    processing row no:  731
    processing row no:  732
    processing row no:  733
    processing row no:  734
    processing row no:  735
    processing row no:  736
    processing row no:  737
    processing row no:  738
    processing row no:  739
    processing row no:  740
    processing row no:  741
    processing row no:  742
    processing row no:  743
    processing row no:  744
    processing row no:  745
    processing row no:  746
    processing row no:  747
    processing row no:  748
    processing row no:  749
    processing row no:  750
    processing row no:  751
    processing row no:  752
    processing row no:  753
    processing row no:  754
    processing row no:  755
    processing row no:  756
    processing row no:  757
    processing row no:  758
    processing row no:  759
    processing row no:  760
    processing row no:  761
    processing row no:  762
    processing row no:  763
    processing row no:  764
    processing row no:  765
    processing row no:  766
    processing row no:  767
    processing row no:  768
    processing row no:  769
    processing row no:  770
    processing row no:  771
    processing row no:  772
    processing row no:  773
    processing row no:  774
    processing row no:  775
    processing row no:  776
    processing row no:  777
    processing row no:  778
    processing row no:  779
    processing row no:  780
    processing row no:  781
    processing row no:  782
    processing row no:  783
    processing row no:  784
    processing row no:  785
    processing row no:  786
    processing row no:  787
    processing row no:  788
    processing row no:  789
    processing row no:  790
    processing row no:  791
    processing row no:  792
    processing row no:  793
    processing row no:  794
    processing row no:  795
    processing row no:  796
    processing row no:  797
    processing row no:  798
    processing row no:  799
    processing row no:  800
    processing row no:  801
    processing row no:  802
    processing row no:  803
    processing row no:  804
    processing row no:  805
    processing row no:  806
    processing row no:  807
    processing row no:  808
    processing row no:  809
    processing row no:  810
    processing row no:  811
    processing row no:  812
    processing row no:  813
    processing row no:  814
    processing row no:  815
    processing row no:  816
    processing row no:  817
    processing row no:  818
    processing row no:  819
    processing row no:  820
    processing row no:  821
    processing row no:  822
    processing row no:  823
    processing row no:  824
    processing row no:  825
    processing row no:  826
    processing row no:  827
    processing row no:  828
    processing row no:  829
    processing row no:  830
    processing row no:  831
    processing row no:  832
    processing row no:  833
    processing row no:  834
    processing row no:  835
    processing row no:  836
    processing row no:  837
    processing row no:  838
    processing row no:  839
    processing row no:  840
    processing row no:  841
    processing row no:  842
    processing row no:  843
    processing row no:  844
    processing row no:  845
    processing row no:  846
    processing row no:  847
    processing row no:  848
    processing row no:  849
    processing row no:  850
    processing row no:  851
    processing row no:  852
    processing row no:  853
    processing row no:  854
    processing row no:  855
    processing row no:  856
    processing row no:  857
    processing row no:  858
    processing row no:  859
    processing row no:  860
    processing row no:  861
    processing row no:  862
    processing row no:  863
    processing row no:  864
    processing row no:  865
    processing row no:  866
    processing row no:  867
    processing row no:  868
    processing row no:  869
    processing row no:  870
    processing row no:  871
    processing row no:  872
    processing row no:  873
    processing row no:  874
    processing row no:  875
    processing row no:  876
    processing row no:  877
    processing row no:  878
    processing row no:  879
    processing row no:  880
    processing row no:  881
    processing row no:  882
    processing row no:  883
    processing row no:  884
    processing row no:  885
    processing row no:  886
    processing row no:  887
    processing row no:  888
    processing row no:  889
    processing row no:  890
    processing row no:  891
    processing row no:  892
    processing row no:  893
    processing row no:  894
    processing row no:  895
    processing row no:  896
    processing row no:  897
    processing row no:  898
    processing row no:  899
    processing row no:  900
    processing row no:  901
    processing row no:  902
    processing row no:  903
    processing row no:  904
    processing row no:  905
    processing row no:  906
    processing row no:  907
    processing row no:  908
    processing row no:  909
    processing row no:  910
    processing row no:  911
    processing row no:  912
    processing row no:  913
    processing row no:  914
    processing row no:  915
    processing row no:  916
    processing row no:  917
    processing row no:  918
    processing row no:  919
    processing row no:  920
    processing row no:  921
    processing row no:  922
    processing row no:  923
    processing row no:  924
    processing row no:  925
    processing row no:  926
    processing row no:  927
    processing row no:  928
    processing row no:  929
    processing row no:  930
    processing row no:  931
    processing row no:  932
    processing row no:  933
    processing row no:  934
    processing row no:  935
    processing row no:  936
    processing row no:  937
    processing row no:  938
    processing row no:  939
    processing row no:  940
    processing row no:  941
    processing row no:  942
    processing row no:  943
    processing row no:  944
    processing row no:  945
    processing row no:  946
    processing row no:  947
    processing row no:  948
    processing row no:  949
    processing row no:  950
    processing row no:  951
    processing row no:  952
    processing row no:  953
    processing row no:  954
    processing row no:  955
    processing row no:  956
    processing row no:  957
    processing row no:  958
    processing row no:  959
    processing row no:  960
    processing row no:  961
    processing row no:  962
    processing row no:  963
    processing row no:  964
    processing row no:  965
    processing row no:  966
    processing row no:  967
    processing row no:  968
    processing row no:  969
    processing row no:  970
    processing row no:  971
    processing row no:  972
    processing row no:  973
    processing row no:  974
    processing row no:  975
    processing row no:  976
    processing row no:  977
    processing row no:  978
    processing row no:  979
    processing row no:  980
    processing row no:  981
    processing row no:  982
    processing row no:  983
    processing row no:  984
    processing row no:  985
    processing row no:  986
    processing row no:  987
    processing row no:  988
    processing row no:  989
    processing row no:  990
    processing row no:  991
    processing row no:  992
    processing row no:  993
    processing row no:  994
    processing row no:  995
    processing row no:  996
    processing row no:  997
    processing row no:  998
    processing row no:  999
    processing row no:  1000



```python
from io import BytesIO 
import requests 
import json 
import pandas as pd 

def put_file(credentials, local_file_name): 
    """This functions returns a StringIO object containing the file content from Bluemix Object Storage V3.""" 
    f = open(local_file_name,'r') 
    my_data = f.read() 
    url1 = ''.join(['https://identity.open.softlayer.com', '/v3/auth/tokens']) 
    data = {'auth': {'identity': {'methods': ['password'], 'password': {'user': {'name': credentials['username'],'domain': {'id': credentials['domain_id']}, 'password': credentials['password']}}}}} 
    headers1 = {'Content-Type': 'application/json'} 
    resp1 = requests.post(url=url1, data=json.dumps(data), headers=headers1) 
    resp1_body = resp1.json() 
    for e1 in resp1_body['token']['catalog']: 
        if(e1['type']=='object-store'): 
            for e2 in e1['endpoints']: 
                if(e2['interface']=='public'and e2['region']== credentials['region']):    
                    url2 = ''.join([e2['url'],'/', credentials['container'], '/', local_file_name]) 
    
    s_subject_token = resp1.headers['x-subject-token'] 
    headers2 = {'X-Auth-Token': s_subject_token, 'accept': 'application/json'} 
    resp2 = requests.put(url=url2, headers=headers2, data = my_data ) 
    print (resp2)
```


```python

df.to_csv('Dataset.csv',index=False)

put_file(credentials_2660 ,'Dataset.csv')
```

    <Response [201]>



```python
prediction = model.predict([get_tf_record('Coca-cola pepsi is so good')])
prediction
```




    array([[9.9977738e-01, 2.2261981e-04]], dtype=float32)


