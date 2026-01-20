# Akıllı Ev Enerji Tüketimi Analizi
Bu projede akıllı ev enerji tüketim verileri üzerinden elektrik tüketimlerini tespit etmeyi amaçlayan bir makine öğrenmesi uygulamasıdır. Projede zaman, hava durumu, geçmiş tüketim verilerini karşılaştırarak tüketim ort üzerinde(anomali) olup olmadığı analiz edilmiştir.

# Projenin Amacı
-Saatlik ve çevresel koşullara göre normal enerji tüketim davranışını analiz etmek

-Bu normal davranıştan sapmaları (anomalileri) otomatik olarak tespit etmek

-Farklı makine öğrenmesi modellerini(KNN,RF,LRegresyon) karşılaştırarak en iyi sonucu veren modeli belirlemek

# Kullanılan Veri Seti
Smart Home Dataset.csv

<img width="594" height="29" alt="image" src="https://github.com/user-attachments/assets/af1e3681-a449-4828-9327-504ac6d26bb4" />

# Kullanılan Kütüphaneler
-pandas

-numpy

-scikit-learn

-matplotlib

-seaborn

# Features
-Zaman

-Sıcaklık

-Nem

-basınç

-Rüzgar
...

# Pivot Kullanımı
Bu projede anomalinin tanımı rasgele yapılmamıştır. Bunun yerine beklenen (baseline) tüketim hesaplanmıştır.

-Oluşturulan Pivot Tabloları-

<img width="349" height="167" alt="image" src="https://github.com/user-attachments/assets/4766097f-ebf7-4e36-a31a-5412c4092e96" />

<img width="380" height="165" alt="image" src="https://github.com/user-attachments/assets/c76d0a87-f8a1-443a-90fe-c27093962af4" />

<img width="391" height="163" alt="image" src="https://github.com/user-attachments/assets/1a106fd2-8e7b-4273-a501-88d275404c67" />

# Baseline Tüketim Hesabı
Verimizdeki her satır için ort tüketimi hesaplamak için:

<img width="268" height="132" alt="image" src="https://github.com/user-attachments/assets/ea22c822-ffb7-478f-9247-7ec532347388" />

Bu şekilde sıcaklık,nem,saat eşit bir şekilde kullanıma etki etmiştir.

# Anomali Tanımı
Tüketim hesaplanan ortalamanın üzerindeyse anomali olmuştur

<img width="714" height="39" alt="image" src="https://github.com/user-attachments/assets/0e02e5ab-cfbd-4aac-b294-86cbca9891ca" />

1-Anomali var Tüketim ort üstünde

0-Normal Tüketim

# Kullanılan Makine Öğrenmesi Modelleri
1-Logistic Regresyon

2-KNN

3-Random Forest

(Lojistik Regresyon ve KNN için StandardScaler kullanılmıştır)

# Model Değerlendirme Metrikleri
-Accuracy (Doğruluk)

-Precision (Hassasiyet)

-Recall (Duyarlılık)

-F1-Score

-Eğitim + Tahmin Süresi


# Modellerin Eğitimleri ve Sonuçları
Veri %80 Eğitim - %20 Test olarak ayırdım. Modeller üzerinde eğittim

<img width="882" height="22" alt="image" src="https://github.com/user-attachments/assets/d2906028-c9d3-4d81-b99d-20e9fde43271" />

-Logisctic Regresyon-

<img width="501" height="375" alt="o1" src="https://github.com/user-attachments/assets/439fde4d-3853-491f-bb61-417d4e42adf1" />

%72.37 Doğruluk ile modeller arasında en kötü model seçildi.

-KNN-

<img width="492" height="328" alt="o2" src="https://github.com/user-attachments/assets/f5ea1616-34e8-4885-b4f1-509d13fa0322" />

%79.09 ile orta iyi seçildi.

-Random Forest-

<img width="487" height="332" alt="o3" src="https://github.com/user-attachments/assets/cda213cc-d678-47fb-8227-275bcf88f9e2" />

%81.33 ile modeller arasında en başarılı sonucu RF verdi.

<img width="467" height="67" alt="o4" src="https://github.com/user-attachments/assets/6c99ca7c-7bd6-4610-80f8-9dab7cc6d20b" />








