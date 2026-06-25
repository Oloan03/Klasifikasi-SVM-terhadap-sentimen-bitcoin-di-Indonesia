# Klasifikasi Sentimen Terhadap Bitcoin Menggunakan algoritma SVM 

Penenitian klasifikasi sentimen twitter di Indonesia terhadap Bitcoin dengan menggunakan algoritma SVM dengan pelabelan otomatis _lexicon-based_ yaitu _Inset Lexicon_ dan _SentisrengthId Lexicon_

## Ringkasan
Penelitian ini melakukan penelitian klasifikasi terhadap dataset yang dikumpulkan dengan alat _tweet-harvest_ yaitu teknik _crawling_ di sosial media _Twitter_ yang sekarang dikenal sebagai _X_.
Data diambil dengan kata kunci dan rentang waktu tertentu.
Proses pelabelan data dilakukan secara otomatis dengan dua basis leksikon berbeda (_Inset & SentisrengthId_).

Pemodel SVM dilakukan dengan alur Pipeline yang berisi proses TF-IDF lalu model SVM dengan tiga kernel berbeda.
