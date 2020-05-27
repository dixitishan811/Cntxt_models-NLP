## Dataset Used :

[Annotated Corpus for Named Entity Recognition using GMB(Groningen Meaning Bank)](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)

**List of entities:**

1. geo = Geographical Entity
2. org = Organization
3. per = Person
4. gpe = Geopolitical Entity
5. tim = Time indicator
6. art = Artifact
7. eve = Event
8. nat = Natural Phenomenon

-----------------------------------------------------------------------

# Results

## 1.CRF :
**Used 5 fold CV and RandomSearchCV to optimize the hyperparameters.**

![Image of Reuslt](https://github.com/ishan-ml/Cntxt_models-NLP/blob/master/NER(Named%20Entity%20Recognition)/Capture.PNG)

## 2.spaCy(Custom Trained) :
These are the results after 30 iterations :

**Precision: 76.77419354838709, Recall: 71.25748502994011, F1 Score: 73.91304347826086**

*These are the overall scores,label wise score can be found in the notebook.*
