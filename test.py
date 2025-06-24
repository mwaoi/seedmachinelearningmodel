import pandas as pd

df = pd.read_csv('Extracted_MALDI_Features_SAMPLE.csv', encoding ='latin-1')
print(df.head())

X = df[['marker_2593', 'marker_2563', 'marker_2503']]
y = df['Species_Label']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)    

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
rf.score(X_test, y_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

features = pd.DataFrame(rf.feature_importances_, index = X.columns)

features.head(15)