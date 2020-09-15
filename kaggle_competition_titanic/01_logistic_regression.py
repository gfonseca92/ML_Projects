import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def unify_dummies(df, column_name):
    column_lenght = len(df.columns)
    df[column_name] = np.nan
    for line in range(0, len(df)):
        for id in range(0, column_lenght):
            if df.iloc[line, id] == 1:
                df.iloc[line, column_lenght] = int(id)
    df_unique = df[column_name]
    return df_unique


def get_numeric_ticket_column(df, df_class):
    for line in range(0, len(df)):
        try:
            df.iloc[line] = float(df.iloc[line])
        except:
            line_value = df.iloc[line].split()
            for i in range(0, len(line_value)):
                try:
                    if float(line_value[i]) > 200:
                        df.iloc[line] = float(line_value[i])
                        break
                except:
                    df.iloc[line] = float(df_class.iloc[line]*100000)
    return df


def treat_data(train_or_test):
    df = pd.read_csv(train_or_test + '.csv')
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    # df['Age_Fare'] = df['Age']*df['Fare']
    df = df.drop(labels=['PassengerId', 'Name', 'Cabin'], axis=1)
    df.dropna(subset=['Embarked'], inplace=True)

    ticket = get_numeric_ticket_column(df['Ticket'], df['Pclass'])

    sex = pd.get_dummies(df['Sex'])
    sex = unify_dummies(sex, 'Sex')
    embarked = pd.get_dummies(df['Embarked'])
    embarked = unify_dummies(embarked, 'Embarked')

    df = df.drop(labels=['Ticket', 'Sex', 'Embarked'], axis=1)
    df = pd.concat([df, sex, embarked, ticket], axis=1)
    sns.pairplot(df, hue='Survived')
    plt.show()
    for col in df.columns:
        is_na = df[col].isna().sum()
        print(str(col) + ' nº of NaNs: ' + str(is_na))
    # df = df.dropna()
    return df


passengerIds = pd.read_csv('test.csv')
passengerIds = pd.DataFrame(passengerIds['PassengerId'], index=None, columns=['PassengerId'])

train = treat_data('train')
test = treat_data('test')
y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
X_test = test

logReg = LogisticRegression()
logReg.fit(X_train, y_train)
predictions = logReg.predict(X_test).astype(int)
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat([passengerIds, predictions], axis=1)
predictions.set_index('PassengerId', inplace=True)
predictions.to_csv('predictions.csv')