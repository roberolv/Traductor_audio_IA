import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings



config = dotenv_values('.env')

ELEVENLABS_API_KEY = 'sk_579f3271c586e5f8062548e1808f681921edc4b67182e2d3'

def translator(audio_file):
    
    # 1.- Trasncribir a texto usando whisper

    try:
        model = whisper.load_model('base')
        result = model.transcribe(audio_file, language='Spanish')
        transcription = result('text')
    except Exception as e:
        gr.Error(f'Se ha producido un error transcribiendo el texto:{str(e)}')
    
    # 2.- Traducir el texto
    try:
        en_transcription = Translator(to_lang='en').translate(transcription)
    except Exception as e:
        raise gr.Error(
            f'Se ha producido un error traduciendo el texto:{str(e)}')
   
    
    # 3.- Generar audio traducido usando elevenlabs (only 10 min free / month)
     
    
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    response = client.text_to_speech.convert(
        voice_id="ODq5zmih8GrVes37Dizd",  # Patrick voice
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=en_transcription,
        model_id="eleven_turbo_v2",  # use the turbo model for low latency, for other languages use the `eleven_multilingual_v2`
        voice_settings= VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    save_file_path = 'audios/en.mp3'

    with open(save_file_path, 'wb') as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    return save_file_path


web= gr.Interface(
    fn=translator,
    inputs=gr.Audio(
        sources=['microphone'],
        type='filepath'
    ),
    outputs=[gr.Audio()],
    title='Traductor de voz a otros idiomas',
    description='Traductor de voz basada en IA a otros idiomas'

)

web.launch()