{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 텍스트 정규화"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import fetch_20newsgroups\n",
    "\n",
    "news_data = fetch_20newsgroups(subset='all',random_state=156)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(news_data.keys())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "print('target 클래스의 값과 분포도 \\n',pd.Series(news_data.target).value_counts().sort_index())\n",
    "print('target 클래스의 이름들 \\n',news_data.target_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(news_data.data[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'list'>\n",
      "학습 데이터 크기 11314 , 테스트 데이터 크기 7532\n"
     ]
    }
   ],
   "source": [
    "from sklearn.datasets import fetch_20newsgroups\n",
    "\n",
    "# subset='train'으로 학습용(Train) 데이터만 추출, remove=('headers', 'footers', 'quotes')로 내용만 추출\n",
    "train_news= fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), random_state=156)\n",
    "X_train = train_news.data\n",
    "y_train = train_news.target\n",
    "print(type(X_train))\n",
    "\n",
    "# subset='test'으로 테스트(Test) 데이터만 추출, remove=('headers', 'footers', 'quotes')로 내용만 추출\n",
    "test_news= fetch_20newsgroups(subset='test',remove=('headers', 'footers','quotes'),random_state=156)\n",
    "X_test = test_news.data\n",
    "y_test = test_news.target\n",
    "print('학습 데이터 크기 {0} , 테스트 데이터 크기 {1}'.format(len(train_news.data) , len(test_news.data)))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 피처 벡터화 변환과 머신러닝 모델 학습/예측/평가"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "학습 데이터 Text의 CountVectorizer Shape: (11314, 101631)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "\n",
    "# Count Vectorization으로 feature extraction 변환 수행. \n",
    "cnt_vect = CountVectorizer()\n",
    "cnt_vect.fit(X_train , y_train)\n",
    "X_train_cnt_vect = cnt_vect.transform(X_train)\n",
    "\n",
    "# 학습 데이터로 fit( )된 CountVectorizer를 이용하여 테스트 데이터를 feature extraction 변환 수행. \n",
    "X_test_cnt_vect = cnt_vect.transform(X_test)\n",
    "\n",
    "print('학습 데이터 Text의 CountVectorizer Shape:',X_train_cnt_vect.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CountVectorized Logistic Regression 의 예측 정확도는 0.617\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# LogisticRegression을 이용하여 학습/예측/평가 수행. \n",
    "lr_clf = LogisticRegression()\n",
    "lr_clf.fit(X_train_cnt_vect , y_train)\n",
    "pred = lr_clf.predict(X_test_cnt_vect)\n",
    "print('CountVectorized Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test,pred)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "TF-IDF Logistic Regression 의 예측 정확도는 0.678\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "# TF-IDF Vectorization 적용하여 학습 데이터셋과 테스트 데이터 셋 변환. \n",
    "tfidf_vect = TfidfVectorizer()\n",
    "tfidf_vect.fit(X_train)\n",
    "X_train_tfidf_vect = tfidf_vect.transform(X_train)\n",
    "X_test_tfidf_vect = tfidf_vect.transform(X_test)\n",
    "\n",
    "# LogisticRegression을 이용하여 학습/예측/평가 수행. \n",
    "lr_clf = LogisticRegression()\n",
    "lr_clf.fit(X_train_tfidf_vect , y_train)\n",
    "pred = lr_clf.predict(X_test_tfidf_vect)\n",
    "print('TF-IDF Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test ,pred)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "TF-IDF Vectorized Logistic Regression 의 예측 정확도는 0.690\n"
     ]
    }
   ],
   "source": [
    "# stop words 필터링을 추가하고 ngram을 기본(1,1)에서 (1,2)로 변경하여 Feature Vectorization 적용.\n",
    "tfidf_vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=300 )\n",
    "tfidf_vect.fit(X_train)\n",
    "X_train_tfidf_vect = tfidf_vect.transform(X_train)\n",
    "X_test_tfidf_vect = tfidf_vect.transform(X_test)\n",
    "\n",
    "lr_clf = LogisticRegression()\n",
    "lr_clf.fit(X_train_tfidf_vect , y_train)\n",
    "pred = lr_clf.predict(X_test_tfidf_vect)\n",
    "print('TF-IDF Vectorized Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test ,pred)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 5 candidates, totalling 15 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=1)]: Done  15 out of  15 | elapsed:  4.0min finished\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression best C parameter : {'C': 10}\n",
      "TF-IDF Vectorized Logistic Regression 의 예측 정확도는 0.704\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "# 최적 C 값 도출 튜닝 수행. CV는 3 Fold셋으로 설정. \n",
    "params = { 'C':[0.01, 0.1, 1, 5, 10]}\n",
    "grid_cv_lr = GridSearchCV(lr_clf ,param_grid=params , cv=3 , scoring='accuracy' , verbose=1 )\n",
    "grid_cv_lr.fit(X_train_tfidf_vect , y_train)\n",
    "print('Logistic Regression best C parameter :',grid_cv_lr.best_params_ )\n",
    "\n",
    "# 최적 C 값으로 학습된 grid_cv로 예측 수행하고 정확도 평가. \n",
    "pred = grid_cv_lr.predict(X_test_tfidf_vect)\n",
    "print('TF-IDF Vectorized Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test ,pred)))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 사이킷런 파이프라인(Pipeline) 사용 및 GridSearchCV와의 결합"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pipeline을 통한 Logistic Regression 의 예측 정확도는 0.704\n"
     ]
    }
   ],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "# TfidfVectorizer 객체를 tfidf_vect 객체명으로, LogisticRegression객체를 lr_clf 객체명으로 생성하는 Pipeline생성\n",
    "pipeline = Pipeline([\n",
    "    ('tfidf_vect', TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=300)),\n",
    "    ('lr_clf', LogisticRegression(C=10))\n",
    "])\n",
    "\n",
    "# 별도의 TfidfVectorizer객체의 fit_transform( )과 LogisticRegression의 fit(), predict( )가 필요 없음. \n",
    "# pipeline의 fit( ) 과 predict( ) 만으로 한꺼번에 Feature Vectorization과 ML 학습/예측이 가능. \n",
    "pipeline.fit(X_train, y_train)\n",
    "pred = pipeline.predict(X_test)\n",
    "print('Pipeline을 통한 Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test ,pred)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 27 candidates, totalling 81 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=1)]: Done  81 out of  81 | elapsed: 32.6min finished\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'lr_clf__C': 10, 'tfidf_vect__max_df': 700, 'tfidf_vect__ngram_range': (1, 2)} 0.755524129397207\n",
      "Pipeline을 통한 Logistic Regression 의 예측 정확도는 0.702\n"
     ]
    }
   ],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "pipeline = Pipeline([\n",
    "    ('tfidf_vect', TfidfVectorizer(stop_words='english')),\n",
    "    ('lr_clf', LogisticRegression())\n",
    "])\n",
    "\n",
    "# Pipeline에 기술된 각각의 객체 변수에 언더바(_)2개를 연달아 붙여 GridSearchCV에 사용될 \n",
    "# 파라미터/하이퍼 파라미터 이름과 값을 설정. . \n",
    "params = { 'tfidf_vect__ngram_range': [(1,1), (1,2), (1,3)],\n",
    "           'tfidf_vect__max_df': [100, 300, 700],\n",
    "           'lr_clf__C': [1,5,10]\n",
    "}\n",
    "\n",
    "# GridSearchCV의 생성자에 Estimator가 아닌 Pipeline 객체 입력\n",
    "grid_cv_pipe = GridSearchCV(pipeline, param_grid=params, cv=3 , scoring='accuracy',verbose=1)\n",
    "grid_cv_pipe.fit(X_train , y_train)\n",
    "print(grid_cv_pipe.best_params_ , grid_cv_pipe.best_score_)\n",
    "\n",
    "pred = grid_cv_pipe.predict(X_test)\n",
    "print('Pipeline을 통한 Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test ,pred)))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
