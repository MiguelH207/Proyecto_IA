import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Paso 1: Cargar el dataset
df = pd.read_csv('emotions.csv')

# Paso 2: Preprocesamiento de texto
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenización y eliminación de stopwords
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    # Lematización
    lemmatized_text = ' '.join([lemmatizer.lemmatize(token) for token in filtered_tokens])
    return lemmatized_text

df['content_processed'] = df['content'].apply(preprocess_text)

# Paso 3: División en conjuntos de entrenamiento y prueba
X = df['content_processed']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Extracción de características (TF-IDF)
vectorizer = TfidfVectorizer(max_features=1000)  # Ajustar según el tamaño del dataset
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Paso 5: Entrenamiento del modelo (SVM)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Paso 6: Evaluación del modelo
y_pred = svm_model.predict(X_test_tfidf)
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Ejemplo de uso del modelo entrenado
def predict_sentiment(text):
    text_processed = preprocess_text(text)
    text_tfidf = vectorizer.transform([text_processed])
    prediction = svm_model.predict(text_tfidf)
    return prediction[0]

# Ejemplos de predicciones
ejemplo1 = "I love this product, it's amazing!"
ejemplo2 = "This movie was terrible, I hated it."
ejemplo3 = "After hours of trying to solve this problem, I'm feeling frustrated with the lack of progress."
ejemplo4 = "I can't wait to start my new job next week! The opportunity to learn and grow in this company has me feeling really excited."

print(f"\nEjemplo 1: '{ejemplo1}' -> Sentimiento: {predict_sentiment(ejemplo1)}")
print(f"Ejemplo 2: '{ejemplo2}' -> Sentimiento: {predict_sentiment(ejemplo2)}")
print(f"Ejemplo 3: '{ejemplo3}' -> Sentimiento: {predict_sentiment(ejemplo3)}")
print(f"Ejemplo 4: '{ejemplo4}' -> Sentimiento: {predict_sentiment(ejemplo4)}")
