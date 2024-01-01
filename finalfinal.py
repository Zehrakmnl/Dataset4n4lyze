import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample

file_path = "./data.xlsx"  
df = pd.read_excel(file_path)

# RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)

# GradientBoostingClassifier 
model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

# ön işlemeden önce
plt.figure(figsize=(10, 6))
sns.histplot(df['Fiyat'], bins=20, kde=False)
plt.title('Fiyat Dağılımı (önişlemesiz)')
plt.xlabel('Fiyat')
plt.show()

# Veriyi ön işleme
def preprocessing(data):
    # Kategorik sütunları sayısal değerlere dönüştürme işlemi
    label_encoder = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = data[categorical_columns].apply(label_encoder.fit_transform)

    return data

# Veriyi ön işleem fonksiyonunu uygula
df = preprocessing(df)

# Veriyi eğitim ve test setlerine ayırıyoruz
x = df.drop('Fiyat', axis=1)
y = df['Fiyat']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Modeli tanımla ve eğit
model = RandomForestClassifier()
model.fit(x_train, y_train)

# -performans-
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

df_majority = df[df['Fiyat'] == 0]
# 'Fiyat' sütununda sayma işlemleri yaparak azınlık sınıfını bulma
minority_class = df['Fiyat'].value_counts().idxmin()
df_minority = df[df['Fiyat'] == minority_class]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])


#################hesaplamalar sonrası çıktılaar
print("Eğitim verisi performansı:" + str(accuracy) + "\n")

print("Kayıt Sayısı:", len(df))

print("Nitelik Sayısı:", len(df.columns))

print("Nitelik Tipleri: ", "\n", df.dtypes, "\n")

print("Merkezi Eğilim:", "\n", df.describe(), "\n")

five_number_summary = df.quantile([0, 0.25, 0.5, 0.75, 1])
print ("5 sayı özeti: ", "\n", five_number_summary)

cv_scores = cross_val_score(model, x_train, y_train, cv=10)
print("Çapraz Doğrulama Skorları:", cv_scores)

# Burada özelliklerin önemlilik durumları ölçülüyor
feature_importance = model.feature_importances_
for feature, importance in zip(x_train.columns, feature_importance):
    print(f"{feature}: {importance}")

# Modeli kaydet
with open("train_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
    
########Görselleştirmeceler
plt.figure(figsize=(10, 6))
sns.histplot(df['Fiyat'], bins=20, kde=False)
plt.title('Fiyat Dağılımı(önişlemeli)')
plt.xlabel('Fiyat')
plt.show()
###########################
df['Yeni Ozellik'] = df['Bellek Hızı'] * df['SSD Kapasitesi']
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Yeni Ozellik', y='Ekran Boyutu', data=df)
plt.title('Bellek Hızı*SSD Kapasitesi ve Ekran Boyutu İlişkisi')
plt.show()

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Özelliklerin Dağılımı (Boxplot)")
plt.show()

# # QQ plot
# plt.figure(figsize=(10, 6))
sm.qqplot(df, line='s')
plt.title("QQ Plot")
plt.show()

