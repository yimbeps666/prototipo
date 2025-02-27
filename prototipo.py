import sounddevice as sd
import numpy as np
import whisper  # Whisper de OpenAI
import ollama
import pyttsx3  # Para text-to-speech

# Configuración de Whisper de OpenAI
whisper_model = whisper.load_model("base")  # Usa el modelo base de Whisper

# Configuración de Ollama y Mistral
model_name = "mistral"  # Nombre del modelo en Ollama

# Configuración de pyttsx3 para voz femenina
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)  # Selecciona una voz femenina (puede variar según el sistema)

# Nombre del asistente
nombre_asistente = "ELISA"

def grabar_audio(duracion=5, tasa_muestreo=16000):
    """Graba audio desde el micrófono."""
    print("Grabando...")
    grabacion = sd.rec(int(duracion * tasa_muestreo), samplerate=tasa_muestreo, channels=1, dtype='float32')
    sd.wait()  # Esperar hasta que la grabación termine
    print("Grabación terminada.")
    return grabacion.flatten()

def transcribir_audio(audio, tasa_muestreo=16000):
    """Transcribe el audio a texto utilizando Whisper de OpenAI."""
    # Asegurar que el audio esté en formato float32
    audio = audio.astype(np.float32)
    
    # Pasar el audio directamente a Whisper
    resultado = whisper_model.transcribe(audio)
    
    return resultado["text"]

def generar_respuesta(texto):
    """Genera una respuesta utilizando Mistral a través de Ollama."""
    # Personalizar el prompt para ELISA
    prompt = (
        f"Eres {nombre_asistente}, un asistente virtual amigable, servicial y con un toque de humor. "
        f"Responde al siguiente mensaje de manera clara y concisa: {texto}"
    )
    respuesta = ollama.generate(model=model_name, prompt=prompt)
    return respuesta["response"]

def hablar(texto):
    """Convierte el texto en voz femenina."""
    print(f"{nombre_asistente}: {texto}")  # Ahora el asistente se identifica con su nombre
    engine.say(texto)
    engine.runAndWait()

def asistente_virtual():
    print(f"{nombre_asistente}: Hola, soy {nombre_asistente}, tu asistente virtual. ¿En qué puedo ayudarte hoy?")
    while True:
        try:
            # Grabar audio
            audio = grabar_audio()
            
            # Transcribir audio a texto
            texto = transcribir_audio(audio)
            print(f"Usuario: {texto}")
            
            # Generar respuesta
            respuesta = generar_respuesta(texto)
            
            # Convertir respuesta a voz
            hablar(respuesta)
            
        except KeyboardInterrupt:
            print(f"{nombre_asistente}: Hasta luego. ¡Fue un placer ayudarte!")
            break

if __name__ == "__main__":
    asistente_virtual()
