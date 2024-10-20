# مدل طبقه‌بندی در SVM

## مقدمه

در این پروژه، هدف استفاده از الگوریتم SVM برای طبقه‌بندی پوکیمون‌ها بر اساس ویژگی‌های مختلف است. طبقه‌بندی پوکیمون‌ها به کلاس های «Standard» و «Not Standard» می‌تواند به کاربران در انتخاب و بهبود استراتژی‌های بازی کمک کند. استفاده از مدل‌های یادگیری ماشین به خصوص SVM به دلیل دقت و توانایی در طبقه‌بندی داده‌های پیچیده، اهمیت بالایی دارد.

**اهمیت و ضرورت مسئله:** در دنیای بازی‌های موبایلی مانند Pokémon Go، شناخت و طبقه‌بندی صحیح پوکیمون‌ها از لحاظ قدرت و کارایی می‌تواند نقش بسزایی در موفقیت بازیکنان ایفا کند. با توجه به حجم بالای داده‌ها و پیچیدگی روابط میان ویژگی‌های مختلف پوکیمون‌ها، استفاده از الگوریتم‌های ماشین یادگیری برای استخراج دانش از این داده‌ها ضروری به نظر می‌رسد.

**دیتاست:** دیتاست مورد استفاده در این پروژه شامل ویژگی‌های مختلف پوکیمون‌هاست که به شناسایی آن‌ها کمک می‌کند. این ویژگی‌ها شامل مواردی نظیر نوع پوکیمون(type)، قدرت حمله(base_attack)، دفاع(base_defence) و سایر ویژگی‌های مرتبط است که برای ساخت مدل SVM استفاده شده‌اند.

## مدل SVM و هایپرپارامترها

الگوریتم SVM یکی از الگوریتم‌های معروف در زمینه یادگیری نظارت شده است که هدف آن یافتن بهترین مرز تصمیم‌گیری برای جدا کردن کلاس‌های مختلف داده‌ها است. در این روش، داده‌ها به صورت نقاطی در فضای چندبعدی نمایش داده می‌شوند و الگوریتم سعی می‌کند مرزی را پیدا کند که حداکثر فاصله بین دو دسته داده را ایجاد کند.

ابرپارامتر های مهم در SVM عبارتند از:

- **پارامتر C:** این ابرپارامتر میزان جریمه داده‌های نادرست طبقه‌بندی شده را کنترل می‌کند. هرچه مقدار C بیشتر باشد، مدل تمایل به کاهش خطاهای آموزش پیدا می‌کند.
- **پارامتر Gamma:** این پارامتر بر شعاع تأثیر داده‌های پشتیبان در فضای ویژگی‌ها تأثیر می‌گذارد. هرچه مقدار Gamma بیشتر باشد، مدل حساس‌تر به داده‌های محلی می‌شود.
- **پارامتر Kernel:** هسته یا Kernel تعیین‌کننده نوع تغییرات و نحوه کارکرد الگوریتم است. رایج‌ترین نوع آن‌ها عبارتند از خطی، چندجمله‌ای و گاوسی (RBF).

## روند مدلسازی SVC

### پیدا کردن داده های از دست رفته (Missing Values)

در ابتدای کار با دستور ‍‍`df.isna().sum()` تعداد داده های از دست رفته در هر ستون را بدست می آوریم. نتایج حاصل شده به شرح زیر است:

```text
pokemon_id                        0
pokemon_name                      0
base_attack                       0
base_defense                      0
base_stamina                      0
type                              0
rarity                            0
charged_moves                     0
fast_moves                        0
candy_required                  536
distance                          0
max_cp                            0
attack_probability              103
base_capture_rate               103
base_flee_rate                  103
dodge_probability               103
max_pokemon_action_frequency    103
min_pokemon_action_frequency    103
found_egg                       263
found_evolution                 263
found_wild                      263
found_research                  263
found_raid                      263
found_photobomb                 263
```

ستون های زیر را بر اساس میانه داده های موجود پر میکنیم. دلیل این امر و استفاده نکردن از میانگین یا مد داده‌ها این است که میانه، در مقایسه با میانگین، کمتر تحت تاثیر داده های پرت قرار میگیرد. استفاده از مد نیز ممکن است ویژگی های داده های کلاس استاندارد به داده های کلاس های دیگر انتقال یابد.

```python
# We used `Median` insted of `Mean` because outliers have less impacts on median insted of mean

df['base_capture_rate'].fillna(df['base_capture_rate'].median(), inplace=True)
df['base_flee_rate'].fillna(df['base_flee_rate'].median(), inplace=True)

df['dodge_probability'].fillna(df['dodge_probability'].median(), inplace=True)
df['attack_probability'].fillna(df['attack_probability'].median(), inplace=True)

df['max_pokemon_action_frequency'].fillna(df['max_pokemon_action_frequency'].median(), inplace=True)
df['min_pokemon_action_frequency'].fillna(df['min_pokemon_action_frequency'].median(), inplace=True)
```

مقادیر از دست رفته ستون `candy_required` را برابر `0` قرار میدهیم. مقادیر از دست رفته ستون های `found_*` را برابر `False` قرار میدهیم.

```python
df['candy_required'].fillna(0, inplace=True)

df['found_evolution'].fillna(False, inplace=True)
df['found_research'].fillna(False, inplace=True)
df['found_raid'].fillna(False, inplace=True)
df['found_photobomb'].fillna(False, inplace=True)
df['found_wild'].fillna(False, inplace=True)
df['found_egg'].fillna(False, inplace=True)
```

### مهندسی ویژگی

ستون های `type` و `charged_moves` و `fast_moves` از نوع دسته بندی هستند. این ستون هارو با One-Hot رمزگذاری میکنیم و در نهایت ستون ها را حذف میکنیم.

```python
# One-Hot Encoding `type`, `charged_moves` and `fast_moves` columns

df['type'] = df['type'].astype(str).apply(lambda x: ast.literal_eval(x))
df['charged_moves'] = df['charged_moves'].astype(str).apply(lambda x: ast.literal_eval(x))
df['fast_moves'] = df['fast_moves'].astype(str).apply(lambda x: ast.literal_eval(x))

cols = ['type', 'charged_moves', 'fast_moves']
for col in cols:
    mlb = MultiLabelBinarizer()
    encoded_df = pd.DataFrame(mlb.fit_transform(df[col]), columns=[f"{col}_{cls}" for cls in mlb.classes_])
    df = pd.concat([df, encoded_df], axis=1)

df.drop(cols, axis=1, inplace=True)
```

### ساخت ستون هدف (Target)

با دستور `df['rarity'].value_counts()` میتوان مقادیر ستون `rarity` را شمرد. نتیجه به شکل زیر است:

```text
rarity
Standard       910
Legendary       64
Mythic          22
Ultra beast     11
```

تعداد پوکیمون های دسته `Standard` روی هم رفته بیشتر از سایر دسته‌ها است. پوکیمون ها را به دو دسته `Standard` و `Not Standard` تقسیم میکنیم تا برای مدلسازی SVM نیز صرفا به دسته بند دو دسته اقدام کنیم.

نمودار های پراکندگی پوکیمون ها بر اساس سه ویژگی پایه `base_attack` و `base_defense` و `base_stamina` مشاهده میکنیم.

![Scatter Plot by Target Class (base attack and defence)](base_a_d.png "Scatter Plot by Target Class (base attack and defence)")

![Scatter Plot by Target Class (base attack and stamina)](base_a_s.png "Scatter Plot by Target Class (base attack and stamina)")

![Scatter Plot by Target Class (base stamina and defence)](base_d_s.png "Scatter Plot by Target Class (base stamina and defence)")

در زیر نیز این سه ویژگی را در یک نمودار سه بعدی میبینیم:

![3D Scatter Plot by Target Class](base_3d.png "3D Scatter Plot by Target Class")

برای نمایش بهتر این ۳ ویژگی در کنار هم از PCA استفاده میکنیم. نمودار حاصل به شکل زیر است:

![Scatter Plot by Standard X Not-Standard Classes](pca_all.png "Scatter Plot by Standard X Not-Standard Classes")

### مدلسازی

مقادیر موجود در ستون target را رمز گذاری میکنیم یدین صورت که مقادیر Standard به مقدار 1 و سایر به 0 تغییر میکند.

پیش از مدلسازی، میزان همبستگی فیچرها را با ستون Target میسنجیم. نمودار حاصل برای همبستگی های بیشتر از 0.1 به صورت زیر است:

![Correlation plot of features with Target](corr.png "Correlation plot of features with Target")

همانطور که مشاهده میشود مقدار distance همبستگی منفی را دارد.

در نهایت دیتافریم ما دارای 274 ویژگی و 1007 رکورد سطر است.

مدل SVC را باپارامتر های ابتدایی زیر آموزش میدهیم و مقادیر test را پیش‌بینی میکنیم:

```python
from sklearn.svm import SVC

model = SVC(kernel='linear' , C=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## نتایج حاصل شده

نتایج حاصل شده از مدلسازی اولیه به شرح زیر است:

```text
Accuracy: 100.00%
====================
Confusion Matrix:
[[ 18   0]
 [  0 184]]
```

نتایج در قالب جدول:

|       | precision | recall | f1-score | support |
| -----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| | | | | |
| accuracy | 1.00 | 1.00 | 1.00 | 1.00 |
| macro avg | 1.00 | 1.00 | 1.00 | 1.00 |
| weighted avg | 1.00 | 1.00 | 1.00 | 1.00 |

در نهایت با استفاده از `GridSearchCV` بهترین پارامتر های مدل برای این مسئله را پیدا میکنیم:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

y_pred_grid = grid.predict(X_test)
print("Accuracy with GridSearch:", accuracy_score(y_test, y_pred_grid))
```

خلاصه نتیجه حاصل شده به شرح زیر است:

```text
Fitting 5 folds for each of 20 candidates, totalling 100 fits
[CV 1/5] END .............C=0.01, kernel=linear;, score=0.994 total time=   2.0s
[CV 2/5] END .............C=0.01, kernel=linear;, score=0.981 total time=   9.4s
[CV 3/5] END .............C=0.01, kernel=linear;, score=0.988 total time=   1.6s
[CV 4/5] END .............C=0.01, kernel=linear;, score=0.981 total time=   2.7s
[CV 5/5] END .............C=0.01, kernel=linear;, score=0.963 total time=   2.4s
[CV 1/5] END ................C=0.01, kernel=rbf;, score=0.907 total time=   0.0s
[CV 2/5] END ................C=0.01, kernel=rbf;, score=0.901 total time=   0.0s
[CV 3/5] END ................C=0.01, kernel=rbf;, score=0.901 total time=   0.0s
[CV 4/5] END ................C=0.01, kernel=rbf;, score=0.901 total time=   0.0s
[CV 5/5] END ................C=0.01, kernel=rbf;, score=0.901 total time=   0.0s
[CV 1/5] END ...............C=0.01, kernel=poly;, score=0.907 total time=   0.0s
[CV 2/5] END ...............C=0.01, kernel=poly;, score=0.913 total time=   0.0s
[CV 3/5] END ...............C=0.01, kernel=poly;, score=0.907 total time=   0.0s
[CV 4/5] END ...............C=0.01, kernel=poly;, score=0.907 total time=   0.0s
[CV 5/5] END ...............C=0.01, kernel=poly;, score=0.901 total time=   0.0s
[CV 1/5] END ............C=0.01, kernel=sigmoid;, score=0.907 total time=   0.0s
[CV 2/5] END ............C=0.01, kernel=sigmoid;, score=0.901 total time=   0.0s
[CV 3/5] END ............C=0.01, kernel=sigmoid;, score=0.901 total time=   0.0s
[CV 4/5] END ............C=0.01, kernel=sigmoid;, score=0.901 total time=   0.0s
[CV 5/5] END ............C=0.01, kernel=sigmoid;, score=0.901 total time=   0.0s
[CV 1/5] END ..............C=0.1, kernel=linear;, score=1.000 total time=  11.3s
[CV 2/5] END ..............C=0.1, kernel=linear;, score=1.000 total time=   3.6s
[CV 3/5] END ..............C=0.1, kernel=linear;, score=0.994 total time=   1.9s
...
[CV 3/5] END .............C=100, kernel=sigmoid;, score=0.851 total time=   0.0s
[CV 4/5] END .............C=100, kernel=sigmoid;, score=0.801 total time=   0.0s
[CV 5/5] END .............C=100, kernel=sigmoid;, score=0.795 total time=   0.0s

Best Parameters: {'C': 0.1, 'kernel': 'linear'}
Accuracy with GridSearch: 1.0
```

---

پس از تنظیم ابرپارامترهای SVM و انجام آزمایش‌های مختلف، بهترین مدل با استفاده از هسته Linear و مقدار 0.1 برای C به دست آمد. این مدل توانست با دقت 100% پوکیمون‌ها را به درستی طبقه‌بندی کند.

ماتریس سردرگمی (Confusion Matrix) نشان داد که مدل توانایی بالایی در تشخیص پوکیمون‌های استاندارد دارد. همچنین، امتیاز F1 نشان داد که مدل توانایی متعادلی در میان کلاس‌های مختلف دارد.

نتایج نهایی نشان می‌دهد که استفاده از الگوریتم SVM می‌تواند به طور مؤثری در طبقه‌بندی پوکیمون‌ها مورد استفاده قرار گیرد و می‌تواند به بهبود استراتژی‌های بازیکنان Pokémon Go کمک کند.
